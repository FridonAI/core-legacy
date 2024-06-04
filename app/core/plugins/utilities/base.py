from typing import Any

from pydantic.v1 import BaseModel


class BaseUtility(BaseModel):
    name: str
    description: str

    def run(self, *args, **kwargs) -> dict | str | Any: ...

    def _run(self, *args, **kwargs) -> str: ...

    async def arun(self, *args, **kwargs) -> dict | str | Any: ...

    async def _arun(self, *args, **kwargs) -> dict | str: ...
