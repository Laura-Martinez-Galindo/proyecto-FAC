#!/usr/bin/env bash
# Muestra que hay libre en Hypatia y por que tus jobs estan en PD.
# Es liviano (solo consulta a Slurm), se puede correr en el nodo de login:
#   bash jobs/recursos_hypatia.sh            # resumen general
#   bash jobs/recursos_hypatia.sh 791388     # ademas, detalle de ese job

echo "=================== PARTICIONES ==================="
echo "CPUS = asignadas/libres/otras/total   (las 'libres' son las que importan)"
sinfo -o "%-12P %.5a %.12l %.6D %.20C %.8m %20G" 2>/dev/null

echo
echo "=================== NODOS CON CPUS LIBRES ==================="
sinfo -N -h -o "%P %N %t %C %e %G" 2>/dev/null | awk '{split($4,c,"/"); if (c[2]>0) printf "%-10s %-14s %-6s libres=%-4s memlibre=%sMB %s\n",$1,$2,$3,c[2],$5,$6}' | sort -k1,1 -k4,4r | head -40

echo
echo "=================== GPUS EN USO POR NODO ==================="
squeue -h -t R -o "%N %b" 2>/dev/null | awk '$2 ~ /gpu/' | sort | uniq -c | head -30

echo
echo "=================== COLA POR PARTICION (R=corriendo, PD=esperando) ==================="
squeue -h -o "%P %t" 2>/dev/null | sort | uniq -c | sort -k2

echo
echo "=================== MIS JOBS ==================="
squeue -u "$USER" -o "%.10i %.10P %.22j %.3t %.10M %.10l %.6C %.8m %.20b %R" 2>/dev/null
echo
echo "Inicio estimado de mis jobs pendientes:"
squeue -u "$USER" -t PD --start -o "%.10i %.10P %.20S %.30R" 2>/dev/null

echo
echo "=================== LIMITES DE CADA PARTICION ==================="
scontrol show partition 2>/dev/null | awk '/PartitionName=/{p=$1} /MaxTime=/{for(i=1;i<=NF;i++) if($i ~ /MaxTime|DefMemPerCPU|MaxNodes/) printf "%s %s ", p, $i; print ""}' | sed 's/PartitionName=//' | sort -u

if [ -n "$1" ]; then
  echo
  echo "=================== DETALLE DEL JOB $1 ==================="
  scontrol show job "$1" 2>/dev/null | grep -E "JobState|Reason|Partition|TimeLimit|NumCPUs|MinMemory|TRES|StartTime|SubmitTime"
  echo "Prioridad (componentes):"
  sprio -j "$1" 2>/dev/null
fi

cat <<'TXT'

Como leerlo:
 * Reason=Priority  -> hay otros jobs con mas prioridad esperando en ESA particion.
 * Reason=Resources -> no hay nodo con lo que pediste (CPUs, memoria o GPU) libre ahora.
 * Pedir MENOS tiempo (--time) ayuda mucho: Slurm "rellena huecos" (backfill) con jobs cortos.
 * Para mover un job que esta en PD a otra particion o bajarle el tiempo, sin reenviarlo:
     scontrol update JobId=<ID> Partition=<particion> TimeLimit=00:45:00
 * O cancelarlo y reenviarlo eligiendo particion/tiempo desde la linea de comando
   (esto pisa lo que diga el #SBATCH del archivo):
     scancel <ID>
     sbatch --partition=<particion> --time=00:45:00 --cpus-per-task=8 --mem=16G jobs/<job>.sbatch
TXT
