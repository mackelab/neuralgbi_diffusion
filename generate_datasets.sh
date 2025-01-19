python -m gbi_diff generate-data --dataset-type two_moons --size 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type SIR --size 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type inverse_kinematics --size 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type lotka_volterra --size 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type gaussian_mixture --size 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type linear_gaussian --size 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type uniform --size 10000 1000 100 --path data

python -m gbi_diff generate-data --dataset-type two_moons --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type SIR --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type inverse_kinematics --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type lotka_volterra --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type gaussian_mixture --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type linear_gaussian --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type uniform --size 10 --path data/observed_data

