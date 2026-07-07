# photextra_pipeline

Pipeline de fotometría multibanda + espectroscopía para galaxias en fusión
(cúmulos CHANCES). Descarga cutouts (GALEX/Legacy/WISE), separa componentes
en pares/sistemas en fusión (xdebpair), mide fotometría por componente o del
sistema total, ajusta espectros DESI/SDSS (continuo estelar + líneas de
emisión vía XpectraFit), y combina fotometría+espectroscopía en un solo SED
por target — con CIGALE (ajuste espectroscópico nativo) o con el método
propio de normalización de apertura.

```
photextra
```

corre el pipeline con un banner de bienvenida, resumen de la corrida y barra
de progreso en la terminal.

## Instalación

```bash
git clone <este repo>
cd photextra_pipeline
pip install -e .
```

Dependencias: `numpy`, `scipy`, `astropy`, `matplotlib`, `pyyaml`, `sep`,
`reproject`, `astroquery`, `photutils`, `rich`.

Además, dos paquetes hermanos (no en PyPI, deben estar clonados aparte y
accesibles vía `sys.path`/`pip install -e`):

- [`xdebpair`](../xdebpair) — separación de fuentes en pares en fusión.
- [`xpectrafit`](../xpectrafit) — ajuste espectral (pPXF + E-MILES, AGN,
  líneas de emisión, BPT/WHAN).

Para el modo `both_method: cigale` (ajuste SED con CIGALE usando el espectro
completo, no solo fotometría de banda ancha) hace falta además el entorno
`cigale` con el fork `cigale_spec` — ver
[`docs/cigale_tutorial/TUTORIAL.pdf`](docs/cigale_tutorial/TUTORIAL.pdf) para
la instalación completa, cómo se registran los filtros propios (CHANCES
continuum + DECam/Legacy reales) y cómo se conecta con este pipeline.

## Uso rápido (CLI)

```bash
photextra config_xmask.yaml --targets mis_targets.csv --limit 5
```

`mis_targets.csv` necesita columnas `ID,RA,DEC,REDSHIFT` (columna opcional
`type` con `merger`/`pre_merger`/`post_merger` — ver `separation` abajo).

## Uso como librería

```python
from photextra_pipeline import Pipeline

pipe = Pipeline(config="config_xmask.yaml", mode="both")
pipe.run({"id": "12345", "ra": 180.1, "dec": 12.3, "z": 0.03})
```

## Configuración (`config_xmask.yaml`)

```yaml
mode: photometry            # photometry | spectroscopy | both
both_method: own            # own | cigale (solo aplica con mode: both)

photometry:
  aperture_mode: mask       # mask (xdebpair) | aperture (circular) | sep_apertures (elipses SEP)
  separation: pair          # central | total | pair
  aperture_radius_arcsec: 5.0

spectroscopy:
  fit_agn: true
  spectrum_fwhm: 2.5
  survey_filters: [Legacy, SDSS]

use_xdebpair: true
surveys: [GALEX_FUV, GALEX_NUV, Legacy_g, Legacy_r, Legacy_z, WISE_W1, WISE_W2, WISE_W3, WISE_W4]
download_size: 1
common_grid: {reference: WISE_W4, pixscale: 1.375, size: 45}
output_dir: ./test_output
```

Los kwargs explícitos al construir `Pipeline(...)` (`mode=`, `use_xdebpair=`,
`both_method=`) pisan lo que diga el YAML — así los scripts existentes
(`run_mkw8_full.py`, etc.) siguen funcionando sin cambios.

### `photometry.aperture_mode`

- **`mask`** (default): separación por componente con `xdebpair` (fallback a
  `xmask` si no está disponible, y a un stub interno de un solo componente si
  tampoco).
- **`aperture`**: apertura circular simple.
- **`sep_apertures`**: elipses de Kron a partir de la detección de fuentes de
  `SEP` (sin `xdebpair`, alternativa rápida sin máscaras).

### `photometry.separation`

Solo aplica con `aperture_mode: mask`. Siempre corre `xdebpair` primero;
después, según el valor:

- **`pair`**: separa por componente, pero la clasificación manda si existe
  una columna `type` en el CSV de targets — `merger`/`pre_merger` → separa;
  `post_merger`/otro → fuerza a 1 componente aunque `xdebpair` haya detectado
  2. Sin columna `type` → se respeta la decisión cruda de `xdebpair`.
- **`central`**: siempre 1 componente (la galaxia principal); si había una
  compañera, se descarta su flujo pero queda un flag
  `has_companion_not_measured=True` + warning en el log.
- **`total`**: une todos los componentes detectados en una sola apertura
  (flujo total del sistema, sin descomponer).

### `both_method`

- **`own`** (default): normalización propia (apertura total vs fibra,
  ponderada por S/N² sobre bandas Legacy, extensión de Zou et al. 2024).
- **`cigale`**: corre CIGALE con el espectro completo (`use_spectro=True`) en
  vez del ajuste propio — ver el tutorial de CIGALE para la instalación.

## Líneas de emisión (espectroscopía)

`spectral_fit.py` extrae, por línea, `flux`, `flux_err`, `ew` (ancho
equivalente), `ew_err`, `sigma` (dispersión de velocidad, km/s), `v_kms`
(corrimiento) y `snr` — para las 22 líneas angostas + 6 líneas anchas (AGN)
que XpectraFit soporta (antes solo se extraían 4 líneas con flux nada más).
XpectraFit salta el ajuste de líneas para espectros muy quiescentes
(EW(Hα)<3Å) a propósito — no es un bug, es una heurística de velocidad.

## Estructura del repo

```
photextra_pipeline/       # paquete principal
  pipeline.py              # orquestación (Pipeline.run)
  downloader.py            # descarga de cutouts (GALEX/Legacy/WISE)
  deblending.py            # adaptador xdebpair
  deblend_photometry.py    # fotometría por componente / xmask fallback
  photometry.py            # medición de flujos (máscara/apertura/sep)
  convolution.py            # PSF matching entre bandas
  reprojection.py          # WCS común + reproyección
  spectrum_acquisition.py  # descarga/cacheo de espectros DESI/SDSS
  spectral_fit.py          # ajuste espectral (XpectraFit) + líneas
  spec_normalization.py    # normalización apertura fibra->total
  cigale_integration.py    # puente hacia CIGALE (both_method: cigale)
  output_table.py          # tabla de salida final
  validation_plots.py      # plots de diagnóstico
  cli.py                    # comando `photextra`
scripts/                   # utilidades CIGALE (filtros, conversión DESI->FITS)
docs/cigale_tutorial/      # instalación CIGALE + filtros + PDF
test_*.py                  # tests (pytest)
```

## Tests

```bash
python -m pytest test_new_features.py -q
```

23 tests (fotometría en los 3 `aperture_mode`, `separation` × clasificación,
bug de máscaras, alineación WCS entre surveys, extracción de líneas
espectrales con espectros reales cacheados, smoke test de `both_method:
cigale`) — todos verdes.
