python -m gbi_diff generate-data --dataset-type two_moons --size 1000000 100000 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type SIR --size 1000000 100000 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type inverse_kinematics --size 1000000 100000 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type lotka_volterra --size 1000000 100000 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type gaussian_mixture --size 1000000 100000 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type linear_gaussian --size 1000000 100000 10000 1000 100 --path data
python -m gbi_diff generate-data --dataset-type uniform --size 1000000 100000 10000 1000 100 --path data

python -m gbi_diff generate-data --dataset-type two_moons --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type SIR --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type inverse_kinematics --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type lotka_volterra --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type gaussian_mixture --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type linear_gaussian --size 10 --path data/observed_data
python -m gbi_diff generate-data --dataset-type uniform --size 10 --path data/observed_data

