# MetaQuant Forge

Aplicación independiente de optimización binaria con modelos Meta Llama, CPU clásica y GPU NVIDIA bajo demanda. Interfaz azul con textos blancos y escenarios de festivales, recarga, restauración de hábitats, logística y sensores. No usa hardware cuántico.

**Empieza por [INSTALACION.txt](INSTALACION.txt)**: GitHub, Modal, Fireworks, Render, dominio y créditos, paso a paso.

## Arquitectura

Render sirve la interfaz y una API ligera. Llama (Fireworks) convierte descripciones en especificaciones validadas y explica resultados medidos. Modal ejecuta CPU y GPU en procesos separados dentro de un contenedor con NVIDIA L4. No hay un servidor GPU permanente ni un endpoint GPU público.

Los ejemplos se ejecutan sin llamar al modelo; formular texto libre y explicar resultados sí consumen API. Los ejemplos grandes son sintéticos, con nuevas semillas y los mismos tamaños, formulaciones matemáticas y presupuestos que la base. Los datos reales necesitan coeficientes explícitos. Las restricciones no soportadas se rechazan; una formulación del modelo debe revisarse antes de ejecutarla.

Conserva tres repeticiones, resultados parciales, mediana y suma de tiempos, verificación independiente de energía y exportación JSON/CSV. GPU obligatoria: no sustituye CUDA por CPU. Los presupuestos de las heurísticas son diferentes; el cociente de tiempos no demuestra ventaja cuántica ni equivalencia de calidad. La latencia de red y arranque no forma parte del tiempo de solver.

## Desarrollo

Python 3.12. Instala `requirements-dev.txt`; ejecuta `python -m pytest -q`. Para vista local: `COOKIE_SECURE=0 python -m uvicorn server.main:app --reload`. Sin credenciales puedes ver la interfaz y validar JSON; la ejecución GPU y Llama permanecen desactivados.

`modal_app.py` contiene las dependencias GPU. `requirements.txt` instala solo el gateway. Despliega el worker con `python -m modal deploy modal_app.py` antes de configurar Render. No instales Torch en Render.

## Alcance operativo

Prototipo para demostraciones de un solo proceso y una sola instancia Render. Solo una comparación simultánea por gateway y un contenedor GPU. Trabajos y sesiones son transitorios: descarga los resultados; un reinicio pierde su seguimiento. Modal impone un máximo de 600 segundos a la función. Un arranque en frío puede tardar más que la estimación de la interfaz.

La web es pública y aplica límites básicos, no autenticación de usuarios ni una cuota monetaria garantizada. Para uso público sostenido añade acceso autenticado, cuotas persistentes, una cola durable y almacenamiento. Configura alertas/límites de gasto en los proveedores antes de difundirla. Un contenedor puede continuar hasta su límite si se interrumpe abruptamente el gateway.

Los textos y datos enviados a Llama pasan por Fireworks; los problemas ejecutados pasan por Modal. Las claves permanecen en servidor. No introduzcas datos confidenciales en la demo.

## Verificación de esta entrega

Ver `VALIDACION.txt`. Las pruebas locales no sustituyen un despliegue real con credenciales. No se ha comprado el dominio ni se han desplegado servicios. No implica admisión ni cumplimiento de una convocatoria de hackathon.
