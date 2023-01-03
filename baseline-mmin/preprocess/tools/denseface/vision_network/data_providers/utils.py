from .cifar import Cifar10DataProvider, Cifar100DataProvider, \
	Cifar10AugmentedDataProvider, Cifar100AugmentedDataProvider
from .svhn import SVHNDataProvider
from .fer import FERPlusDataProvider, AVECDataProvider, MUSEDataProvider, VGGFACE2DataProvieder


def get_data_provider_by_name(name, data_dir, train_params):
	"""Return required data provider class"""
	if name == 'C10':
		return Cifar10DataProvider(save_path=data_dir, **train_params)
	if name == 'C10+':
		return Cifar10AugmentedDataProvider(save_path=data_dir, **train_params)
	if name == 'C100':
		return Cifar100DataProvider(save_path=data_dir, **train_params)
	if name == 'C100+':
		return Cifar100AugmentedDataProvider(save_path=data_dir, **train_params)
	if name == 'SVHN':
		return SVHNDataProvider(**train_params)
	if name == 'FER+':
		return FERPlusDataProvider(data_dir, **train_params)
	if name == 'AVEC':
		return AVECDataProvider(data_dir, **train_params)
	if name == 'MUSE':
		return MUSEDataProvider(data_dir, **train_params)
	if name == 'VGGFACE2':
		return VGGFACE2DataProvieder(data_dir, **train_params)
	else:
		print("Sorry, data provider for `%s` dataset "
			  "was not implemented yet" % name)
		exit()
