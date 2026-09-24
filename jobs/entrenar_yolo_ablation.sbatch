#!/bin/bash
#SBATCH --job-name=train_yolo_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/logs/train_yolo_ablation_%j.out
#SBATCH --error=jobs/logs/train_yolo_ablation_%j.err

set -e
mkdir -p jobs/logs pesos runs/detect

echo "=========================================================="
echo "Iniciando Entrenamiento de YOLOv11 en Ramas de Ablación"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

# 1. Entrenar en Rama 2: Sin HUD
echo ""
echo ">>> [1/3] Entrenando YOLOv11 en Rama: Sin HUD..."
python scripts/entrenar_yolo_ablation.py \
    --yaml-dataset datasets/dataset_ablation_final/2_sin_hud/dataset_2_sin_hud.yaml \
    --nombre-exp yolov11_sin_hud \
    --modelo-base yolo11n.pt \
    --epocas 100 \
    --batch 32 \
    --device 0,1

# 2. Entrenar en Rama 3: UDVD Estándar
echo ""
echo ">>> [2/3] Entrenando YOLOv11 en Rama: Denoised UDVD Estándar..."
python scripts/entrenar_yolo_ablation.py \
    --yaml-dataset datasets/dataset_ablation_final/3_denoised_udvd_standard/dataset_3_udvd_standard.yaml \
    --nombre-exp yolov11_udvd_std \
    --modelo-base yolo11n.pt \
    --epocas 100 \
    --batch 32 \
    --device 0,1

# 3. Entrenar en Rama 4: UDVD Mejorado (Destriping + UDVD)
echo ""
echo ">>> [3/3] Entrenando YOLOv11 en Rama: Denoised UDVD Mejorado..."
python scripts/entrenar_yolo_ablation.py \
    --yaml-dataset datasets/dataset_ablation_final/4_denoised_udvd_mejorado/dataset_4_udvd_mejorado.yaml \
    --nombre-exp yolov11_udvd_mejorado \
    --modelo-base yolo11n.pt \
    --epocas 100 \
    --batch 32 \
    --device 0,1

# 4. Evaluación comparativa completa de instancias y mAP
echo ""
echo ">>> [4/4] Ejecutando Matriz de Evaluación Final con Conteo de Instancias..."
python scripts/evaluar_ablation_yolo_completo.py --device 0

echo "=========================================================="
echo "Entrenamientos y Evaluación de Ablación finalizados exitosamente."
echo "Fecha: $(date)"
echo "=========================================================="
