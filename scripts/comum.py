# -*- coding: utf-8 -*-
"""
comum.py - utilitarios compartilhados por todas as etapas do pipeline.

Cada etapa (01_*.py ... 08_*.py) segue o mesmo contrato:
  * le  <entrada>/manifesto.csv  (gerado pela etapa anterior);
  * grava as imagens processadas em <saida>/imagens/ (e mascaras, quando houver);
  * grava <saida>/manifesto.csv  (lido pela etapa seguinte);
  * grava <saida>/erros.csv      (imagens que falharam - o lote NAO para);
  * grava <saida>/etapa.log      (log detalhado do que foi processado);
  * grava <saida>/resumo.json    (contagens, parametros e tempo de execucao);
  * opcionalmente grava <saida>/exemplos/ (paineis "antes | depois" p/ o artigo).

Os caminhos dentro do manifesto sao RELATIVOS a pasta do proprio manifesto, de
modo que a pasta de saidas pode ser movida inteira sem quebrar as referencias.

Nenhuma funcao aqui altera a imagem: somente leitura/escrita, log e organizacao.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # barra de progresso; se o tqdm faltar, o pipeline continua funcionando
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iteravel, **_kwargs):  # type: ignore
        return iteravel

# Colunas do manifesto que guardam caminhos de arquivo.
CAMPOS_CAMINHO = ("imagem", "mascara", "fov", "pelos", "mascara_pred")
NOME_MANIFESTO = "manifesto.csv"


# --------------------------------------------------------------------------- #
# Leitura e escrita de imagens (segura para caminhos com acento no Windows)
# --------------------------------------------------------------------------- #
def ler_imagem(caminho: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Le uma imagem com cv2.imdecode (cv2.imread falha com acentos no Windows)."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        raise ValueError(f"Arquivo vazio: {caminho}")
    img = cv2.imdecode(dados, flags)
    if img is None:
        raise ValueError(f"Nao foi possivel decodificar a imagem: {caminho}")
    return img


def ler_mascara(caminho: Path) -> np.ndarray:
    """Le uma mascara binaria e devolve uint8 com valores {0, 255}."""
    m = ler_imagem(caminho, cv2.IMREAD_GRAYSCALE)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def salvar_imagem(caminho: Path, img: np.ndarray, qualidade_jpg: int = 95) -> None:
    """Grava a imagem (PNG sem perdas por padrao) criando as pastas necessarias."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ext = caminho.suffix.lower()
    params: List[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(qualidade_jpg)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Falha ao codificar a imagem para {caminho}")
    buf.tofile(str(caminho))


# --------------------------------------------------------------------------- #
# Manifesto (CSV) e caminhos relativos
# --------------------------------------------------------------------------- #
def caminho_relativo(destino: Path, base: Path) -> str:
    """Caminho de 'destino' relativo a 'base' (absoluto se estiverem em discos diferentes)."""
    destino = Path(destino).resolve()
    try:
        return Path(os.path.relpath(destino, Path(base).resolve())).as_posix()
    except ValueError:  # Windows: unidades diferentes (C: x D:)
        return destino.as_posix()


def resolver_caminho(valor: str, base: Path) -> Optional[Path]:
    """Converte o valor do manifesto (relativo a 'base') em caminho absoluto."""
    if valor is None or str(valor).strip() == "":
        return None
    p = Path(valor)
    return p if p.is_absolute() else (Path(base) / p).resolve()


def ler_manifesto(pasta: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Le <pasta>/manifesto.csv. Encerra com mensagem clara se nao existir."""
    arq = Path(pasta) / NOME_MANIFESTO
    if not arq.is_file():
        raise SystemExit(
            f"[ERRO] Manifesto nao encontrado em '{arq}'.\n"
            f"       Rode a etapa anterior primeiro ou confira o parametro --entrada."
        )
    with open(arq, "r", encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        linhas = [dict(l) for l in leitor]
        campos = list(leitor.fieldnames or [])
    return linhas, campos


def escrever_csv(arq: Path, linhas: Sequence[Dict], campos: Optional[Sequence[str]] = None) -> None:
    """Grava uma lista de dicionarios em CSV (UTF-8). Une as chaves de todas as linhas."""
    arq = Path(arq)
    arq.parent.mkdir(parents=True, exist_ok=True)
    ordem: List[str] = list(campos or [])
    for l in linhas:
        for k in l.keys():
            if k not in ordem:
                ordem.append(k)
    with open(arq, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordem, extrasaction="ignore")
        w.writeheader()
        for l in linhas:
            w.writerow({k: ("" if l.get(k) is None else l.get(k)) for k in ordem})


def salvar_json(arq: Path, dados: Dict) -> None:
    Path(arq).parent.mkdir(parents=True, exist_ok=True)
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Log
# --------------------------------------------------------------------------- #
def configurar_log(saida: Path, nome: str, verboso: bool = False) -> logging.Logger:
    """Log em arquivo (<saida>/etapa.log, sempre detalhado) e no console (resumido)."""
    Path(saida).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(nome)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(Path(saida) / "etapa.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verboso else logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# --------------------------------------------------------------------------- #
# Exemplos "antes | depois" para as figuras do artigo
# --------------------------------------------------------------------------- #
def escolher_exemplos(ids: Iterable[str], n: int, semente: int) -> set:
    """Sorteio deterministico: com a mesma semente, as etapas escolhem as MESMAS imagens."""
    ids = sorted(set(ids))
    if n <= 0 or not ids:
        return set()
    rng = random.Random(semente)
    rng.shuffle(ids)
    return set(ids[:n])


def _para_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def montar_painel(paineis: Sequence[Tuple[str, np.ndarray]], altura: int = 320) -> np.ndarray:
    """Coloca imagens lado a lado (mesma altura) com um titulo em cima de cada uma."""
    blocos = []
    for titulo, img in paineis:
        img = _para_bgr(img)
        h, w = img.shape[:2]
        nova_w = max(1, int(round(w * altura / float(h))))
        img = cv2.resize(img, (nova_w, altura), interpolation=cv2.INTER_AREA)
        faixa = np.full((34, nova_w, 3), 255, np.uint8)
        cv2.putText(faixa, titulo[:40], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        blocos.append(np.vstack([faixa, img]))
        blocos.append(np.full((altura + 34, 6, 3), 255, np.uint8))  # espacador
    return np.hstack(blocos[:-1])


def salvar_exemplo(saida: Path, image_id: str, paineis: Sequence[Tuple[str, np.ndarray]]) -> None:
    salvar_imagem(Path(saida) / "exemplos" / f"{image_id}_antes_depois.png", montar_painel(paineis))


# --------------------------------------------------------------------------- #
# Argumentos padrao de linha de comando
# --------------------------------------------------------------------------- #
def criar_parser(descricao: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=descricao,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--entrada", required=True, help="Pasta de entrada (saida da etapa anterior).")
    p.add_argument("--saida", required=True, help="Pasta de saida desta etapa.")
    p.add_argument("--exemplos", type=int, default=0,
                   help="Quantidade de paineis antes/depois a salvar em <saida>/exemplos (0 = nenhum).")
    p.add_argument("--semente", type=int, default=42, help="Semente para a escolha dos exemplos.")
    p.add_argument("--limite", type=int, default=0,
                   help="Processa so as N primeiras imagens (teste rapido). 0 = todas.")
    p.add_argument("--formato", choices=["png", "jpg"], default="png",
                   help="Formato das imagens gravadas (png = sem perdas, recomendado).")
    p.add_argument("--verboso", action="store_true", help="Mostra o log detalhado tambem no console.")
    return p


# --------------------------------------------------------------------------- #
# Executor generico de uma etapa
# --------------------------------------------------------------------------- #
@dataclass
class Contexto:
    """Informacoes que a funcao de processamento de cada etapa recebe."""
    entrada: Path
    saida: Path
    logger: logging.Logger
    exemplos: set
    formato: str
    params: Dict = field(default_factory=dict)

    def destino(self, subpasta: str, nome: str, ext: Optional[str] = None) -> Path:
        """Caminho de gravacao dentro de <saida>/<subpasta>/."""
        ext = ext or self.formato
        return self.saida / subpasta / f"{nome}.{ext}"


FuncaoEtapa = Callable[[Dict, Contexto], Dict]


def executar_etapa(nome: str, args: argparse.Namespace, processar: FuncaoEtapa,
                   params: Dict, descricao_resumo: str = "") -> int:
    """
    Roda 'processar' em cada linha do manifesto de entrada.

    - Antes de chamar 'processar', os campos de caminho viram caminhos absolutos (Path).
    - 'processar' devolve a linha atualizada (pode trocar caminhos e adicionar colunas).
    - Qualquer excecao em uma imagem e registrada em erros.csv e o lote continua.
    - Retorna 0 se ao menos uma imagem foi processada; 1 caso contrario.
    """
    entrada, saida = Path(args.entrada), Path(args.saida)
    logger = configurar_log(saida, nome, getattr(args, "verboso", False))
    params_publicos = {k: v for k, v in params.items() if not k.startswith("_")}  # "_x" = uso interno
    logger.info(f"Etapa {nome} | entrada={entrada.resolve()} | saida={saida.resolve()}")
    logger.info(f"Parametros: {json.dumps(params_publicos, ensure_ascii=False, default=str)}")

    linhas, campos = ler_manifesto(entrada)
    if args.limite and args.limite > 0:
        linhas = linhas[: args.limite]
        logger.info(f"--limite ativo: processando apenas {len(linhas)} imagens.")

    ctx = Contexto(entrada=entrada, saida=saida, logger=logger,
                   exemplos=escolher_exemplos([l["image_id"] for l in linhas], args.exemplos, args.semente),
                   formato=args.formato, params=params)

    novas: List[Dict] = []
    erros: List[Dict] = []
    t0 = time.perf_counter()
    for linha in tqdm(linhas, desc=nome, unit="img"):
        iid = linha.get("image_id", "?")
        try:
            trabalho = dict(linha)
            for c in CAMPOS_CAMINHO:  # relativo (manifesto de entrada) -> absoluto
                if c in trabalho:
                    trabalho[c] = resolver_caminho(trabalho[c], entrada)
            t_img = time.perf_counter()
            nova = processar(trabalho, ctx)
            for c in CAMPOS_CAMINHO:  # absoluto -> relativo (manifesto de saida)
                if c in nova:
                    nova[c] = "" if nova[c] is None else caminho_relativo(nova[c], saida)
            nova[f"tempo_{nome}_ms"] = round((time.perf_counter() - t_img) * 1000, 1)
            novas.append(nova)
            logger.debug(f"OK   {iid}")
        except Exception as exc:  # erro em UMA imagem nao derruba o lote
            erros.append({"image_id": iid, "etapa": nome, "erro": f"{type(exc).__name__}: {exc}"})
            logger.warning(f"ERRO {iid}: {type(exc).__name__}: {exc}")
            logger.debug(traceback.format_exc())

    duracao = time.perf_counter() - t0
    escrever_csv(saida / NOME_MANIFESTO, novas, campos)
    escrever_csv(saida / "erros.csv", erros, ["image_id", "etapa", "erro"])
    resumo = {
        "etapa": nome,
        "descricao": descricao_resumo,
        "entrada": str(entrada.resolve()),
        "saida": str(saida.resolve()),
        "parametros": params_publicos,
        "imagens_na_entrada": len(linhas),
        "imagens_ok": len(novas),
        "imagens_com_erro": len(erros),
        "tempo_total_s": round(duracao, 2),
        "tempo_medio_por_imagem_ms": round(1000 * duracao / max(1, len(linhas)), 1),
        "exemplos_salvos": sorted(ctx.exemplos),
    }
    salvar_json(saida / "resumo.json", resumo)
    logger.info(f"Concluido: {len(novas)} ok, {len(erros)} com erro, {duracao:.1f} s. "
                f"Resumo em {saida / 'resumo.json'}")
    if erros:
        logger.info(f"Lista de falhas em {saida / 'erros.csv'}")
    return 0 if novas else 1


# --------------------------------------------------------------------------- #
# Pequenas funcoes de imagem reaproveitadas por varias etapas
# --------------------------------------------------------------------------- #
def mascara_ou_cheia(caminho: Optional[Path], forma: Tuple[int, int]) -> np.ndarray:
    """Le a mascara de campo de visao (FOV); se nao houver, considera a imagem toda valida."""
    if caminho is None:
        return np.full(forma, 255, np.uint8)
    m = ler_mascara(caminho)
    if m.shape[:2] != forma:
        m = cv2.resize(m, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST)
    return m


def sobrepor_contorno(img: np.ndarray, mascara: np.ndarray, cor=(0, 255, 0)) -> np.ndarray:
    """Desenha o contorno de uma mascara binaria sobre a imagem (uso so ilustrativo)."""
    saida = _para_bgr(img).copy()
    contornos, _ = cv2.findContours((mascara > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(saida, contornos, -1, cor, 2)
    return saida
