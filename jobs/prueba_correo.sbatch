#!/usr/bin/env bash

#SBATCH --job-name=prueba_correo
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/prueba_correo_%j.out
#SBATCH --error=jobs/logs/prueba_correo_%j.err

echo "Inicio de la prueba de correo"
sleep 20
echo "Fin de la prueba de correo"
