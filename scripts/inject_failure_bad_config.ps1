# Failure B: break MODEL_PATH in the ConfigMap, then restart so pods pick it up (readiness fails).
kubectl -n mlops-app patch configmap inference-config --type merge -p '{"data":{"MODEL_PATH":"/models/missing.joblib"}}'
kubectl -n mlops-app rollout restart deploy/prediction-api
Write-Host "Injected failure B. Fix with: kubectl apply -f kubernetes/configmap.yaml (then restart via the agent)"
