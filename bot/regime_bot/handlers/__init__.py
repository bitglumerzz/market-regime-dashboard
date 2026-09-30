from aiogram import Router

from . import admin, payments, user


def setup() -> Router:
    """Собрать дерево роутеров. Роутеры — модульные объекты, поэтому перед повторной сборкой
    (новый Dispatcher в тестах) отвязываем их от прежнего корня."""
    root = Router(name="root")
    for r in (admin.router, payments.router, user.router):
        r._parent_router = None
    root.include_routers(admin.router, payments.router, user.router)   # админ первым: перехватывает /stats у админов
    return root
