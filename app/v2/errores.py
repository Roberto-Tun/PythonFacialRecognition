"""Errores de la v2: código estable + mensaje en español, sin el texto de la
excepción interna (la v1 devolvía `str(e)`, que puede filtrar rutas o datos)."""


class ErrorV2(Exception):
    def __init__(self, estado: int, codigo: str, mensaje: str):
        super().__init__(mensaje)
        self.estado = estado
        self.codigo = codigo
        self.mensaje = mensaje

    def cuerpo(self) -> dict:
        return {"error": self.codigo, "mensaje": self.mensaje}
