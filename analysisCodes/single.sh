# python3 hotrgTc.py --chi 30 --isGilt --isSym --Ngilt 2 --legcut 2 --gilteps 6e-6 --maxiter 36 --rootiter 15 --stbk 15 --Thi 1.00001 --Tlow 0.999989
python hotrgFlow.py --chi 30 --Ngilt 2 --legcut 2 --gilteps 6e-6 --maxiter 36 --stbk 15
python drawRGflow.py --chi 30 --isGilt --gilteps 6e-6 --scheme Gilt-HOTRG --Ngilt 2 --legcut 2 --isDiffAcc
python hotrgScale.py --chi 30 --gilteps 6e-6 --Ngilt 2 --legcut 2 --iRGlow 14 --iRGhi 29 >> /Users/brucelyu/paper-ongoing/TensorSpaceRGflow/out/gilt_hotrg22_flow/eps6e-06_chi30/prt_scD.txt
