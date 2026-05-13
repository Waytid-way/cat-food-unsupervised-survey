# Cat Food Unsupervised Survey

Build the deployment artifacts with:

```bash
python scripts/build_deployment_artifacts.py
```

Run the dashboard locally with:

```bash
python -m catfood_unsupervised.dashboard
```

Production starts with:

```bash
gunicorn wsgi:server --bind 0.0.0.0:$PORT
```
