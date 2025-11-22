#$ -S /bin/bash
qsub -l a100=1,s_vmem=200G \
     -N All_target_48h_seed2025 \
     -e /home/wanzhe/OmniTFT/qsub_output/test \
     -o /home/wanzhe/OmniTFT/qsub_output/test \
     qsub_run_script.sh