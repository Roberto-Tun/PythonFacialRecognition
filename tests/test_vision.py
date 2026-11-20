"""T-04: la foto vertical del teléfono llega girada según su EXIF y reducida."""

import io

from PIL import Image

from app.v2.vision import LADO_MAXIMO, leer_imagen


def _jpeg(ancho, alto, orientacion=None) -> bytes:
    img = Image.new("RGB", (ancho, alto), (120, 80, 40))
    salida = io.BytesIO()
    if orientacion is None:
        img.save(salida, "JPEG")
    else:
        exif = Image.Exif()
        exif[0x0112] = orientacion  # Orientation
        img.save(salida, "JPEG", exif=exif.tobytes())
    return salida.getvalue()


def test_gira_segun_exif():
    # El sensor guarda 400×300 y el EXIF dice "girar 90°": la foto real es vertical.
    alto, ancho = leer_imagen(_jpeg(400, 300, orientacion=6)).shape[:2]
    assert (ancho, alto) == (300, 400)


def test_reduce_fotos_grandes():
    alto, ancho = leer_imagen(_jpeg(4000, 3000)).shape[:2]
    assert max(alto, ancho) == LADO_MAXIMO


def test_archivo_que_no_es_imagen():
    import pytest

    from app.v2.errores import ErrorV2

    with pytest.raises(ErrorV2) as e:
        leer_imagen(b"no soy una foto")
    assert e.value.codigo == "imagen_invalida"
