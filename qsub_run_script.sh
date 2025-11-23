#$ -S /bin/bash

# 加载 Conda 配置
# source /home/wanzhe/anaconda3/etc/profile.d/conda.sh

# 激活指定的 Conda 环境
# conda activate tfticu
source /home/wanzhe/anaconda3/bin/activate tf213
# 运行 Python 脚本
python /home/wanzhe/OmniTFT/Train_one_target.py . yes
# python /home/wanzhe/Sepsis_TFT/All_feature_attention.py Sepsis . yes
echo "Script run ended"
# 退出 Conda 环境
conda deactivate

echo "Script executed successfully"

