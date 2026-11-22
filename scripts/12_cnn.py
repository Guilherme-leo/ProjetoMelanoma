# -*- coding: utf-8 -*-
"""
ETAPA 12 (OPCIONAL) - CNN PARA COMPARACAO COM O METODO CLASSICO        [Artigo: Secoes 5.4 e 6.3]

O que faz
  Treina uma rede convolucional pre-treinada no ImageNet (transferencia de
  aprendizado: ResNet-18 por padrao, ou EfficientNet-B0) para classificar
  benigno x melanoma diretamente das imagens padronizadas da etapa 08.
    * saida com 1 neuronio + BCEWithLogitsLoss com peso para a classe positiva
      (compensa o desbalanceamento);
    * aumento de dados com reflexoes e rotacoes de 90 graus (a lesao nao tem
      orientacao "certa");
    * parada antecipada pela AUC no conjunto de VALIDACAO (o melhor modelo e salvo);
    * limiar de decisao escolhido na validacao para atingir a sensibilidade-alvo;
    * mesmas metricas da etapa 11 (metricas.py), para comparar na Tabela 7.

Requisitos
  PyTorch e torchvision (NAO estao no requirements.txt basico - ver README).
  Na primeira execucao com pre-treino, o torchvision baixa os pesos da rede
  (~45 MB para a ResNet-18) da internet.
  Sem GPU, use --congelar (treina so a ultima camada; bem mais rapido em CPU).

Justificativa (topico adicional: Introducao a CNNs aplicadas a classificacao)
  Serve de referencia para avaliar se os descritores classicos (interpretaveis)
  perdem muito desempenho em relacao a uma abordagem de aprendizado profundo.

AVISO: a saida e um ESCORE DE SUSPEITA para apoio a decisao, nao um diagnostico.

Uso
  python scripts/12_cnn.py --entrada saidas/08_padronizacao --saida saidas/12_cnn --congelar
  python scripts/12_cnn.py --entrada saidas/08_padronizacao_visual --saida saidas/12_cnn_visual --epocas 20
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402
import metricas  # noqa: E402

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main() -> int:
    p = comum.criar_parser("Etapa 12 (opcional) - CNN com transferencia de aprendizado.")
    p.add_argument("--arquitetura", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    p.add_argument("--sem-pretreino", action="store_true", help="Treina do zero (nao recomendado com ~2.000 imagens).")
    p.add_argument("--congelar", action="store_true", help="Treina so a camada final (rapido em CPU).")
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0, help="Taxa de aprendizado. 0 = 1e-3 com --congelar, 1e-4 sem.")
    p.add_argument("--paciencia", type=int, default=4, help="Epocas sem melhora da AUC de validacao antes de parar.")
    p.add_argument("--tamanho", type=int, default=224, help="Lado da imagem na entrada da rede.")
    p.add_argument("--normalizacao", choices=["imagenet", "dataset"], default="imagenet",
                   help="imagenet (recomendado com pre-treino) ou media/desvio do normalizacao.json da etapa 08.")
    p.add_argument("--sensibilidade-alvo", type=float, default=0.85)
    p.add_argument("--dispositivo", default="auto", help="auto, cpu ou cuda.")
    args = p.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models
    except ImportError:
        print("[ERRO] PyTorch/torchvision nao instalados. Esta etapa e opcional; veja o README (secao CNN).")
        return 1

    entrada, saida = Path(args.entrada), Path(args.saida)
    logger = comum.configurar_log(saida, "12_cnn", args.verboso)
    random.seed(args.semente)
    np.random.seed(args.semente)
    torch.manual_seed(args.semente)
    disp = torch.device("cuda" if (args.dispositivo == "auto" and torch.cuda.is_available()) else
                        ("cpu" if args.dispositivo == "auto" else args.dispositivo))
    logger.info(f"Dispositivo: {disp}")

    linhas, _ = comum.ler_manifesto(entrada)
    if args.limite:
        linhas = linhas[: args.limite]
    validas, erros = [], []
    for l in linhas:  # confere os arquivos ANTES de treinar (o DataLoader nao pode falhar no meio)
        cam = comum.resolver_caminho(l["imagem"], entrada)
        if cam is None or not cam.is_file():
            erros.append({"image_id": l["image_id"], "etapa": "12_cnn", "erro": "imagem nao encontrada"})
        else:
            l["_caminho"] = cam
            validas.append(l)
    conj = {s: [l for l in validas if l.get("split") == s] for s in ("treino", "validacao", "teste")}
    if not conj["validacao"]:  # sem validacao oficial: separa 15% do treino, estratificado
        rng = random.Random(args.semente)
        pos = [l for l in conj["treino"] if int(l["alvo_melanoma"]) == 1]
        neg = [l for l in conj["treino"] if int(l["alvo_melanoma"]) == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        val = pos[: max(1, len(pos) * 15 // 100)] + neg[: max(1, len(neg) * 15 // 100)]
        ids_val = {l["image_id"] for l in val}
        conj["validacao"] = val
        conj["treino"] = [l for l in conj["treino"] if l["image_id"] not in ids_val]
        logger.warning("Sem conjunto de validacao: 15% do treino foi separado para validacao.")
    logger.info(" | ".join(f"{s}: {len(v)}" for s, v in conj.items()))
    if len(conj["treino"]) < 20:
        logger.error("Treino pequeno demais.")
        return 1

    if args.normalizacao == "dataset" and (entrada / "normalizacao.json").is_file():
        nj = json.loads((entrada / "normalizacao.json").read_text(encoding="utf-8"))
        media, desvio = np.array(nj["media"], np.float32), np.array(nj["desvio"], np.float32)
    else:
        media, desvio = MEDIA_IMAGENET, DESVIO_IMAGENET

    class Lesoes(Dataset):
        def __init__(self, itens: List[Dict], aumentar: bool):
            self.itens, self.aumentar = itens, aumentar

        def __len__(self):
            return len(self.itens)

        def __getitem__(self, i):
            l = self.itens[i]
            img = cv2.cvtColor(comum.ler_imagem(l["_caminho"]), cv2.COLOR_BGR2RGB)
            if img.shape[0] != args.tamanho or img.shape[1] != args.tamanho:
                img = cv2.resize(img, (args.tamanho, args.tamanho), interpolation=cv2.INTER_AREA)
            x = img.astype(np.float32) / 255.0
            if self.aumentar:
                if random.random() < 0.5:
                    x = x[:, ::-1]
                if random.random() < 0.5:
                    x = x[::-1]
                x = np.rot90(x, random.randint(0, 3))
            x = (x - media) / desvio
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
            return x, torch.tensor(float(l["alvo_melanoma"]), dtype=torch.float32)

    def carregador(itens, aumentar, embaralhar):
        return DataLoader(Lesoes(itens, aumentar), batch_size=args.lote, shuffle=embaralhar, num_workers=0)

    # ---- modelo
    pre = not args.sem_pretreino
    if args.arquitetura == "resnet18":
        rede = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pre else None)
        rede.fc = nn.Linear(rede.fc.in_features, 1)
        cabeca = rede.fc
    else:
        rede = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pre else None)
        rede.classifier[1] = nn.Linear(rede.classifier[1].in_features, 1)
        cabeca = rede.classifier[1]
    if args.congelar:
        for prm in rede.parameters():
            prm.requires_grad = False
        for prm in cabeca.parameters():
            prm.requires_grad = True
    rede = rede.to(disp)

    y_tr = np.array([int(l["alvo_melanoma"]) for l in conj["treino"]])
    peso_pos = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
    perda = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_pos], device=disp))
    lr = args.lr or (1e-3 if args.congelar else 1e-4)
    otim = torch.optim.AdamW([q for q in rede.parameters() if q.requires_grad], lr=lr, weight_decay=1e-4)
    logger.info(f"{args.arquitetura} | pre-treino={pre} | congelar={args.congelar} | lr={lr} | peso_pos={peso_pos:.2f}")

    def prever(itens) -> np.ndarray:
        rede.eval()
        saidas = []
        with torch.no_grad():
            for x, _ in carregador(itens, False, False):
                saidas.append(torch.sigmoid(rede(x.to(disp))).squeeze(1).cpu().numpy())
        return np.concatenate(saidas) if saidas else np.zeros(0)

    y_val = np.array([int(l["alvo_melanoma"]) for l in conj["validacao"]])
    historico, melhor_auc, melhor_estado, sem_melhora = [], -1.0, None, 0
    t0 = time.perf_counter()
    for ep in range(1, args.epocas + 1):
        rede.train()
        soma, n = 0.0, 0
        for x, y in comum.tqdm(carregador(conj["treino"], True, True), desc=f"12_cnn epoca {ep}", unit="lote"):
            x, y = x.to(disp), y.to(disp)
            otim.zero_grad()
            loss = perda(rede(x).squeeze(1), y)
            loss.backward()
            otim.step()
            soma += float(loss.item()) * len(y)
            n += len(y)
        prob_val = prever(conj["validacao"])
        auc_val = metricas.calcular(y_val, prob_val, 0.5)["auc"]
        historico.append({"epoca": ep, "perda_treino": round(soma / max(n, 1), 5),
                          "auc_validacao": round(float(auc_val), 5) if np.isfinite(auc_val) else ""})
        logger.info(f"epoca {ep}: perda={soma / max(n, 1):.4f}  AUC validacao={auc_val:.4f}")
        if np.isfinite(auc_val) and auc_val > melhor_auc:
            melhor_auc, melhor_estado, sem_melhora = auc_val, copy.deepcopy(rede.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                logger.info(f"Parada antecipada na epoca {ep}.")
                break
    if melhor_estado is not None:
        rede.load_state_dict(melhor_estado)
    torch.save({"estado": rede.state_dict(), "arquitetura": args.arquitetura, "tamanho": args.tamanho,
                "media": media.tolist(), "desvio": desvio.tolist()}, saida / "modelo_cnn.pt")

    prob_val = prever(conj["validacao"])
    limiar = metricas.escolher_limiar(y_val, prob_val, args.sensibilidade_alvo)
    logger.info(f"Limiar escolhido na validacao: {limiar:.4f}")
    tabela, predicoes, curvas = [], [], {}
    rotulo = f"CNN {args.arquitetura}"
    for split, itens in conj.items():
        if not itens:
            continue
        y = np.array([int(l["alvo_melanoma"]) for l in itens])
        prob = prob_val if split == "validacao" else prever(itens)
        for tipo, lim in (("escolhido", limiar), ("0.5", 0.5)):
            tabela.append(metricas.arredondar({"modelo": rotulo, "conjunto": split, "limiar_tipo": tipo,
                                               **metricas.calcular(y, prob, lim)}))
        curvas[split] = (y, prob)
        for l, yy, pp in zip(itens, y, prob):
            predicoes.append({"modelo": "cnn", "image_id": l["image_id"], "split": split, "alvo_melanoma": int(yy),
                              "escore_suspeita": round(float(pp), 5), "previsto_melanoma": int(pp >= limiar)})
        if split in ("validacao", "teste"):
            metricas.salvar_matriz(metricas.calcular(y, prob, limiar), saida / f"matriz_confusao_cnn_{split}.png",
                                   f"{rotulo} - {split} (limiar {limiar:.3f})")
    metricas.salvar_roc({f"{rotulo} - {s}": c for s, c in curvas.items() if s != "treino"},
                        saida / "roc_cnn.png", "Curva ROC - CNN")
    comum.escrever_csv(saida / "historico_treino.csv", historico)
    comum.escrever_csv(saida / "metricas.csv", tabela)
    comum.escrever_csv(saida / "predicoes.csv", predicoes)
    comum.escrever_csv(saida / "erros.csv", erros, ["image_id", "etapa", "erro"])
    comum.salvar_json(saida / "resumo.json", {
        "etapa": "12_cnn", "entrada": str(entrada.resolve()),
        "aviso": "Escore de suspeita para APOIO a decisao medica; nao e diagnostico.",
        "parametros": {k: v for k, v in vars(args).items()}, "dispositivo": str(disp),
        "n_por_conjunto": {s: len(v) for s, v in conj.items()}, "melhor_auc_validacao": melhor_auc,
        "limiar": limiar, "epocas_executadas": len(historico),
        "tempo_total_s": round(time.perf_counter() - t0, 2),
    })
    for t in tabela:
        if t["limiar_tipo"] == "escolhido":
            logger.info(f"{t['conjunto']:<10} sens={t['sensibilidade']} esp={t['especificidade']} "
                        f"f1={t['f1']} auc={t['auc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
