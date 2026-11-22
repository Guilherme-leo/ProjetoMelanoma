# -*- coding: utf-8 -*-
"""
ETAPA 11 - CLASSIFICACAO DE PADROES: BENIGNO x MELANOMA (CLASSICO)    [Artigo: Secoes 5.4, 5.5 e 6.3]

O que faz
  1. Le descritores.csv (etapa 10). Atributos = colunas do dicionario_descritores.csv.
  2. Para cada modelo (SVM-RBF, Random Forest, Regressao Logistica e um
     classificador trivial "dummy" como referencia minima):
       * pipeline: imputacao pela mediana -> padronizacao (z-score) -> modelo;
       * pesos de classe balanceados (melanoma e minoria, < 20% da base);
       * busca de hiperparametros com validacao cruzada estratificada (5 folds)
         NO TREINO, otimizando a AUC;
       * probabilidades "fora da dobra" (out-of-fold) no treino para escolher o
         LIMIAR que atinge a sensibilidade-alvo (padrao 0,85) com a maior
         especificidade possivel. (--limiar-com validacao usa o conjunto de validacao.)
  3. Avalia em validacao e teste com o limiar FIXADO antes (o teste e usado uma unica vez).
  4. Grava metricas, predicoes, curvas ROC, matrizes de confusao, importancia dos
     atributos (permutacao) e os modelos treinados (.joblib).

Justificativa (topico: Classificacao de Padroes)
  Classificadores classicos sobre descritores interpretaveis (ABCD) permitem
  explicar ao medico POR QUE uma lesao recebeu escore alto. A acuracia nao e
  usada sozinha porque a base e desbalanceada; a metrica prioritaria e a
  sensibilidade, pois o falso negativo e o erro mais grave na triagem.

AVISO: a saida e um ESCORE DE SUSPEITA para apoio a decisao, nao um diagnostico.

Uso
  python scripts/11_classificacao.py --entrada saidas/10_descritores --saida saidas/11_classificacao
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402
import metricas  # noqa: E402

CAMPOS_NAO_ATRIBUTO = {"image_id", "split", "rotulo", "alvo_melanoma", "iou", "dice"}


def ler_tabela(pasta: Path) -> Tuple[List[Dict], List[str]]:
    arq = pasta / "descritores.csv"
    if not arq.is_file():
        raise SystemExit(f"[ERRO] {arq} nao encontrado. Rode a etapa 10 antes.")
    with open(arq, "r", encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    dic = pasta / "dicionario_descritores.csv"
    if dic.is_file():
        with open(dic, "r", encoding="utf-8", newline="") as f:
            atributos = [l["atributo"] for l in csv.DictReader(f)]
    else:
        atributos = [c for c in (linhas[0].keys() if linhas else []) if c not in CAMPOS_NAO_ATRIBUTO]
    return linhas, atributos


def para_matriz(linhas: List[Dict], atributos: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    X = np.array([[num(l.get(a)) for a in atributos] for l in linhas], dtype=float)
    X[~np.isfinite(X)] = np.nan
    y = np.array([int(float(l["alvo_melanoma"])) for l in linhas], dtype=int)
    return X, y, [l["image_id"] for l in linhas]


def construir(nome: str, semente: int):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if nome == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=semente)
        grade = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}
    elif nome == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=semente, n_jobs=-1)
        grade = {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": [1, 3, 5]}
    elif nome == "logreg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        grade = {"clf__C": [0.01, 0.1, 1, 10]}
    elif nome == "dummy":
        clf = DummyClassifier(strategy="stratified", random_state=semente)
        grade = {}
    else:
        raise ValueError(nome)
    pipe = Pipeline([("imputar", SimpleImputer(strategy="median")),
                     ("padronizar", StandardScaler()), ("clf", clf)])
    return pipe, grade


NOMES_MODELOS = {"svm": "SVM (RBF)", "rf": "Random Forest", "logreg": "Regressao Logistica",
                 "dummy": "Referencia trivial (dummy)"}


def main() -> int:
    p = comum.criar_parser("Etapa 11 - classificacao benigno x melanoma com descritores (scikit-learn).")
    p.add_argument("--modelos", nargs="+", choices=list(NOMES_MODELOS), default=list(NOMES_MODELOS))
    p.add_argument("--sensibilidade-alvo", type=float, default=0.85,
                   help="Sensibilidade minima desejada ao escolher o limiar de decisao.")
    p.add_argument("--limiar-com", choices=["oof", "validacao"], default="oof",
                   help="Onde escolher o limiar: predicoes fora-da-dobra do treino (oof) ou conjunto de validacao.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--sem-busca", action="store_true", help="Nao faz busca de hiperparametros (mais rapido).")
    args = p.parse_args()

    try:
        import joblib
        from sklearn.base import clone
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    except ImportError:
        print("[ERRO] scikit-learn nao instalado. Rode: pip install -r requirements.txt")
        return 1

    entrada, saida = Path(args.entrada), Path(args.saida)
    logger = comum.configurar_log(saida, "11_classificacao", args.verboso)
    t0 = time.perf_counter()
    linhas, atributos = ler_tabela(entrada)
    if args.limite:
        linhas = linhas[: args.limite]
    conjuntos = {s: [l for l in linhas if l.get("split") == s] for s in ("treino", "validacao", "teste")}
    logger.info(f"{len(atributos)} atributos | " + " | ".join(f"{s}: {len(v)}" for s, v in conjuntos.items()))
    if len(conjuntos["treino"]) < 20:
        logger.error("Treino com menos de 20 imagens: nao ha como treinar.")
        return 1
    dados = {s: para_matriz(v, atributos) for s, v in conjuntos.items() if v}
    X_tr, y_tr, _ = dados["treino"]
    if len(np.unique(y_tr)) < 2:
        logger.error("O treino precisa ter as duas classes (melanoma e benigno).")
        return 1
    folds = max(2, min(args.folds, int(np.bincount(y_tr).min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.semente)

    tabela, predicoes, erros, resumo_modelos = [], [], [], {}
    curvas = {s: {} for s in dados}
    for nome in args.modelos:
        rotulo = NOMES_MODELOS[nome]
        try:
            logger.info(f"Treinando {rotulo} ...")
            pipe, grade = construir(nome, args.semente)
            if grade and not args.sem_busca:
                busca = GridSearchCV(pipe, grade, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
                busca.fit(X_tr, y_tr)
                modelo, melhores = busca.best_estimator_, busca.best_params_
                logger.info(f"  melhores hiperparametros: {melhores} (AUC CV = {busca.best_score_:.4f})")
            else:
                modelo, melhores = clone(pipe).fit(X_tr, y_tr), {}
            oof = cross_val_predict(clone(modelo), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]

            if args.limiar_com == "validacao" and "validacao" in dados and len(np.unique(dados["validacao"][1])) == 2:
                X_v, y_v, _ = dados["validacao"]
                limiar = metricas.escolher_limiar(y_v, modelo.predict_proba(X_v)[:, 1], args.sensibilidade_alvo)
            else:
                limiar = metricas.escolher_limiar(y_tr, oof, args.sensibilidade_alvo)
            logger.info(f"  limiar escolhido = {limiar:.4f} (sensibilidade-alvo {args.sensibilidade_alvo})")

            for split, (X, y, ids) in dados.items():
                prob = oof if split == "treino" else modelo.predict_proba(X)[:, 1]
                nome_split = "treino (fora-da-dobra)" if split == "treino" else split
                for tipo, lim in (("escolhido", limiar), ("0.5", 0.5)):
                    m = metricas.calcular(y, prob, lim)
                    tabela.append(metricas.arredondar({"modelo": rotulo, "conjunto": nome_split,
                                                       "limiar_tipo": tipo, **m}))
                curvas[split][rotulo] = (y, prob)
                for iid, yy, pp in zip(ids, y, prob):
                    predicoes.append({"modelo": nome, "image_id": iid, "split": split, "alvo_melanoma": int(yy),
                                      "escore_suspeita": round(float(pp), 5), "previsto_melanoma": int(pp >= limiar)})
                if split in ("teste", "validacao") and nome != "dummy":
                    metricas.salvar_matriz(metricas.calcular(y, prob, limiar),
                                           saida / f"matriz_confusao_{nome}_{split}.png",
                                           f"{rotulo} - {split} (limiar {limiar:.3f})")
            joblib.dump({"modelo": modelo, "atributos": atributos, "limiar": limiar,
                         "hiperparametros": melhores}, saida / f"modelo_{nome}.joblib")
            resumo_modelos[nome] = {"hiperparametros": melhores, "limiar": round(limiar, 5),
                                    "auc_oof": round(float(metricas.calcular(y_tr, oof, limiar)["auc"]), 4)}
        except Exception as exc:  # um modelo com problema nao impede os demais
            erros.append({"image_id": "-", "etapa": f"11_classificacao[{nome}]", "erro": f"{type(exc).__name__}: {exc}"})
            logger.exception(f"Falha no modelo {rotulo}: {exc}")

    if not tabela:
        logger.error("Nenhum modelo foi treinado com sucesso.")
        comum.escrever_csv(saida / "erros.csv", erros, ["image_id", "etapa", "erro"])
        return 1

    for split, c in curvas.items():
        metricas.salvar_roc(c, saida / f"roc_{split}.png", f"Curva ROC - {split}")

    # Melhor modelo nao trivial pela AUC fora-da-dobra (escolha feita SEM olhar o teste).
    candidatos = {k: v for k, v in resumo_modelos.items() if k != "dummy"}
    melhor = max(candidatos, key=lambda k: candidatos[k]["auc_oof"]) if candidatos else None
    if melhor:
        alvo_imp = "teste" if "teste" in dados else ("validacao" if "validacao" in dados else "treino")
        X_i, y_i, _ = dados[alvo_imp]
        if len(np.unique(y_i)) == 2:
            mod = joblib.load(saida / f"modelo_{melhor}.joblib")["modelo"]
            imp = permutation_importance(mod, X_i, y_i, scoring="roc_auc", n_repeats=10,
                                         random_state=args.semente, n_jobs=-1)
            ordem = np.argsort(-imp.importances_mean)
            linhas_imp = [{"atributo": atributos[i], "queda_auc_media": round(float(imp.importances_mean[i]), 5),
                           "desvio": round(float(imp.importances_std[i]), 5)} for i in ordem]
            comum.escrever_csv(saida / f"importancia_atributos_{melhor}.csv", linhas_imp)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                top = linhas_imp[:15][::-1]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh([t["atributo"] for t in top], [t["queda_auc_media"] for t in top],
                        xerr=[t["desvio"] for t in top])
                ax.set_xlabel(f"Queda media da AUC ao embaralhar o atributo ({alvo_imp})")
                ax.set_title(f"Importancia por permutacao - {NOMES_MODELOS[melhor]}")
                fig.tight_layout()
                fig.savefig(saida / f"importancia_atributos_{melhor}.png", dpi=200)
                plt.close(fig)
            except ImportError:
                pass

    comum.escrever_csv(saida / "metricas.csv", tabela)
    comum.escrever_csv(saida / "predicoes.csv", predicoes)
    comum.escrever_csv(saida / "erros.csv", erros, ["image_id", "etapa", "erro"])
    comum.salvar_json(saida / "resumo.json", {
        "etapa": "11_classificacao", "entrada": str(entrada.resolve()),
        "aviso": "Escore de suspeita para APOIO a decisao medica; nao e diagnostico.",
        "atributos": atributos, "n_por_conjunto": {s: len(v) for s, v in conjuntos.items()},
        "parametros": {"modelos": args.modelos, "sensibilidade_alvo": args.sensibilidade_alvo,
                       "limiar_com": args.limiar_com, "folds": folds, "busca": not args.sem_busca},
        "modelos": resumo_modelos, "melhor_modelo_por_auc_oof": melhor,
        "tempo_total_s": round(time.perf_counter() - t0, 2),
    })
    for t in tabela:
        if t["limiar_tipo"] == "escolhido" and t["conjunto"] in ("validacao", "teste"):
            logger.info(f"{t['modelo']:<28} {t['conjunto']:<10} sens={t['sensibilidade']} esp={t['especificidade']} "
                        f"f1={t['f1']} auc={t['auc']}")
    logger.info(f"Concluido. Melhor modelo (AUC fora-da-dobra): {melhor}. Resultados em {saida.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
