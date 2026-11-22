import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('constancia_cor', ROOT / 'scripts' / '05_constancia_cor.py')
cor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cor)


class ConstanciaCorTest(unittest.TestCase):
    def test_iluminante_usa_somente_pixels_validos_mesmo_com_amostra_pequena(self):
        img = np.full((20, 20, 3), 255, dtype=np.uint8)
        img[:2, :2] = [40, 80, 160]
        valido = np.zeros((20, 20), dtype=bool)
        valido[:2, :2] = True
        saida, e = cor.shades_of_gray(img, valido, 6)
        np.testing.assert_allclose(e, np.array([40, 80, 160]) / np.linalg.norm([40, 80, 160]))
        np.testing.assert_allclose(saida[0, 0], [106, 106, 106], atol=1)

    def test_sem_pixels_validos_falha_em_vez_de_usar_fundo(self):
        with self.assertRaisesRegex(ValueError, 'nenhum pixel valido'):
            cor.shades_of_gray(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=bool), 6)

    def test_expoente_grande_e_infinito_sao_estaveis(self):
        img = np.array([[[6, 12, 24], [8, 16, 32]]], dtype=np.uint8)
        valido = np.ones((1, 2), dtype=bool)
        for p in (10000, float('inf')):
            _, e = cor.shades_of_gray(img, valido, p)
            np.testing.assert_allclose(e, np.array([1, 2, 4]) / np.sqrt(21), atol=1e-6)

    def test_limites_invalidos_sao_rejeitados(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts' / '05_constancia_cor.py'),
                                 '--entrada', 'x', '--saida', 'y', '--escuro', '250'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--escuro < --saturado', result.stderr)


if __name__ == '__main__':
    unittest.main()
