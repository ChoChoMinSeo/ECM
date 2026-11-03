# Quality / lambda: qs0=0.0018, qs1=0.0035, qs2=0.0067,
#                   qs3=0.0130, qs4=0.0250, qs5=0.0483.
# IMS RD-search lambda is inferred from qs0-qs5 filenames; otherwise add --lmbda.

# Test IMS: omit --mode for image-level RD search, or add --mode 0 ... 5.
python ./eval.py --cuda True --real True \
 --checkpoint ../exp_results/pretrained_models/v016_unfreeze/qs5.pth.tar \
 --data ./data/test/00_Kodak \
 --output ./output/IMS_qs5

# Test BMS: prepare and check manual maps with dataset_links/BMS_dataset/mode_selector/.
python ./eval_bms.py --cuda True --real True \
 --checkpoint ../exp_results/pretrained_models/v312_6modes/qs5.pth.tar \
 --data ./data/BMS/images \
 --mode_map_path ./data/BMS/modes \
 --output ./output/BMS_qs5
