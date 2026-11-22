# -*- coding: utf-8 -*-
"""
juncao.py - executa as etapas do pipeline EM ORDEM, preservando todas as saidas.

Cada etapa le a pasta da etapa anterior e grava na sua propria pasta numerada:

  dados/isic2017  -> 01_inventario -> 02_redimensionamento -> 03_remocao_bordas
  -> 04_remocao_pelos -> 05_constancia_cor -> 06_realce_contraste
  -> 07_filtragem_fourier -> 08_padronizacao -> 09_segmentacao
  -> 10_descritores -> 11_classificacao
  (12_cnn e opcional, le a pasta 08_padronizacao e so roda com --ate 12)

Exemplos
  # tudo, do inicio ao fim, com 6 exemplos antes/depois por etapa
  python scripts/juncao.py --dados dados/isic2017 --saidas saidas --exemplos 6

  # teste rapido com 30 imagens por conjunto
  python scripts/juncao.py --dados dados/isic2017 --saidas saidas_teste --amostra 30 --exemplos 3

  # so um intervalo (a etapa 03 precisa que saidas/02_redimensionamento ja exista)
  python scripts/juncao.py --saidas saidas --de 3 --ate 5

  # parametros especificos de uma etapa (pode repetir --extra)
  python scripts/juncao.py --dados dados/isic2017 --extra "4:--limiar 10 --pelos-claros" --extra "7:--d0 0.06"

O script para na primeira etapa que terminar com erro (codigo de saida != 0),
mas as saidas das etapas anteriores ficam intactas para inspecao.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

PASTA_SCRIPTS = Path(__file__).resolve().parent

# (numero, script, pasta de saida)
ETAPAS = [
    (1, "01_inventario.py", "01_inventario"),
    (2, "02_redimensionamento.py", "02_redimensionamento"),
    (3, "03_remocao_bordas.py", "03_remocao_bordas"),
    (4, "04_remocao_pelos.py", "04_remocao_pelos"),
    (5, "05_constancia_cor.py", "05_constancia_cor"),
    (6, "06_realce_contraste.py", "06_realce_contraste"),
    (7, "07_filtragem_fourier.py", "07_filtragem_fourier"),
    (8, "08_padronizacao.py", "08_padronizacao"),
    (9, "09_segmentacao.py", "09_segmentacao"),
    (10, "10_descritores.py", "10_descritores"),
    (11, "11_classificacao.py", "11_classificacao"),
    (12, "12_cnn.py", "12_cnn"),  # opcional: exige PyTorch; so roda com --ate 12
]
# Etapas que nao leem a etapa imediatamente anterior.
ENTRADA_ESPECIAL = {12: "08_padronizacao"}
ULTIMA_PADRAO = 11


def parse_extras(itens: List[str]) -> Dict[int, List[str]]:
    extras: Dict[int, List[str]] = {}
    for item in itens or []:
        if ":" not in item:
            raise SystemExit(f"--extra invalido: '{item}'. Use o formato 'N:--param valor'.")
        num, resto = item.split(":", 1)
        extras.setdefault(int(num), []).extend(shlex.split(resto))
    return extras


def main() -> int:
    p = argparse.ArgumentParser(description="Executa o pipeline de pre-processamento em sequencia.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dados", default="dados/isic2017", help="Pasta com os arquivos do ISIC 2017 (entrada da etapa 01).")
    p.add_argument("--saidas", default="saidas", help="Pasta onde ficam as pastas numeradas de cada etapa.")
    p.add_argument("--de", type=int, default=1, help="Primeira etapa a executar.")
    p.add_argument("--ate", type=int, default=ULTIMA_PADRAO,
                   help="Ultima etapa a executar (12 = CNN opcional, exige PyTorch).")
    p.add_argument("--exemplos", type=int, default=0, help="Paineis antes/depois por etapa.")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--amostra", type=int, default=0, help="(etapa 01) N imagens por conjunto, estratificado.")
    p.add_argument("--limite", type=int, default=0, help="Processa so as N primeiras imagens em cada etapa.")
    p.add_argument("--formato", choices=["png", "jpg"], default="png")
    p.add_argument("--extra", action="append", default=[],
                   help="Argumentos extras para uma etapa, ex.: --extra \"4:--limiar 10\".")
    p.add_argument("--continuar-se-falhar", action="store_true",
                   help="Nao interrompe a sequencia se uma etapa terminar com erro.")
    p.add_argument("--simular", action="store_true", help="So mostra os comandos, sem executar.")
    args = p.parse_args()

    if not (1 <= args.de <= args.ate <= len(ETAPAS)):
        p.error(f"Intervalo invalido: --de {args.de} --ate {args.ate} (validos: 1..{len(ETAPAS)}).")
    extras = parse_extras(args.extra)
    saidas = Path(args.saidas)

    for num, script, pasta in ETAPAS:
        if num < args.de or num > args.ate:
            continue
        if num == 1:
            entrada = Path(args.dados)
        else:
            entrada = saidas / ENTRADA_ESPECIAL.get(num, ETAPAS[num - 2][2])
        # a etapa 11 le descritores.csv; as demais leem manifesto.csv
        exigido = entrada / ("descritores.csv" if num == 11 else "manifesto.csv")
        if num > 1 and not exigido.is_file() and not args.simular:
            print(f"[ERRO] Etapa {num:02d} precisa de '{exigido}'. Rode antes a etapa que gera essa pasta.")
            return 1
        cmd = [sys.executable, str(PASTA_SCRIPTS / script), "--entrada", str(entrada),
               "--saida", str(saidas / pasta), "--exemplos", str(args.exemplos),
               "--semente", str(args.semente), "--formato", args.formato]
        if args.limite:
            cmd += ["--limite", str(args.limite)]
        if num == 1 and args.amostra:
            cmd += ["--amostra", str(args.amostra)]
        cmd += extras.get(num, [])

        print("\n" + "=" * 78)
        print(f"ETAPA {num:02d} - {script}")
        print("  " + " ".join(shlex.quote(c) for c in cmd))
        print("=" * 78, flush=True)
        if args.simular:
            continue
        t0 = time.perf_counter()
        codigo = subprocess.call(cmd)
        print(f"-> etapa {num:02d} terminou com codigo {codigo} em {time.perf_counter() - t0:.1f} s "
              f"(log: {saidas / pasta / 'etapa.log'})")
        if codigo != 0 and not args.continuar_se_falhar:
            print(f"[PARADO] Corrija a etapa {num:02d} e retome com: --de {num}")
            return codigo
    print("\nPipeline concluido. Confira os resumo.json e erros.csv de cada pasta.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
