from service.service import app, router

# Ensure router is registered with app
if not any(r.path == "/stream" for r in app.routes):
    app.include_router(router)

__all__ = ["app"]
