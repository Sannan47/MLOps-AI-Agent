# Failure A: point the deployment at an image tag that does not exist in the cluster.
kubectl -n mlops-app set image deploy/prediction-api api=mlops-inference:9.9.9-missing
Write-Host "Injected failure A. Revert with: kubectl -n mlops-app rollout undo deploy/prediction-api"
