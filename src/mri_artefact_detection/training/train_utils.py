'''
    Helper classes and functions for model training. 
'''

import os, random, time
import numpy as np, matplotlib.pyplot as plt

import torch, tensorflow as tf
import torchio as tio

from keras.utils import Sequence
from keras.callbacks import Callback
from sklearn.model_selection import GroupShuffleSplit

class PrintModelWeightsNorm(Callback):
    """
    Keras callback to print the L2 norm of the model weights at the end of 
    each epoch.

    Attributes
    ----------
    model : keras.Model
        The Keras model being trained.

    Methods
    -------
    on_epoch_end(epoch, logs=None)
        Called at the end of each epoch to compute and print the weights' norm.
    """

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the end of each epoch during training.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Dictionary of logs from the epoch, defaults to None.
        :type logs: dict, optional
        """
        weights_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(w)) for w in self.model.weights))
        print(f"\nNorm of weights after epoch {epoch+1}: {weights_norm.numpy()}")


def split_by_patient(real_image_paths: list, pids: list, real_labels: list) -> tuple:
    """
    Splits image data by patient ID into training, validation, and test sets.

    :param real_image_paths: Paths to the images.
    :type real_image_paths: list
    :param pids: Corresponding patient IDs for each image.
    :type pids: list
    :param real_labels: Labels for each image.
    :type real_labels: list

    :returns: Tuple containing training, validation, and test sets (Xtrain, Xval, Xtest, ytrain, yval, ytest).
    :rtype: tuple
    """
    # remove patients with only one image
    pid_counts = {pid:0 for pid in pids}
    for pid in pids:
        pid_counts[pid] += 1
    pids_to_remove = [pid for pid in pid_counts if pid_counts[pid] < 3]
    
    real_image_paths = [path for path, pid in zip(real_image_paths, pids) if pid not in pids_to_remove]
    real_labels = [label for label, pid in zip(real_labels, pids) if pid not in pids_to_remove]
    pids = [pid for pid in pids if pid not in pids_to_remove]

    artf_frc = [1,0,0]
    while max(artf_frc)-min(artf_frc)>0.02:
        for trvidx, testidx in GroupShuffleSplit(n_splits=1, test_size=0.1).split(real_image_paths, real_labels, pids):
            Xtrval, Xtest = [real_image_paths[i] for i in trvidx], [real_image_paths[i] for i in testidx]
            ytrval, ytest = [real_labels[i] for i in trvidx], [real_labels[i] for i in testidx]
            pid_trval, pid_test = [pids[i] for i in trvidx], [pids[i] for i in testidx]

        for tridx, validx in GroupShuffleSplit(n_splits=1, test_size=0.2222).split(Xtrval, ytrval, pid_trval):
            Xtrain, Xval = [Xtrval[i] for i in tridx], [Xtrval[i] for i in validx]
            ytrain, yval = [ytrval[i] for i in tridx], [ytrval[i] for i in validx]
            pid_train, pid_val = [pid_trval[i] for i in tridx], [pid_trval[i] for i in validx]

        # assert there is not overlap between pid_train and pid_test, and pid_val
        assert len(set(pid_train).intersection(set(pid_test))) == 0
        assert len(set(pid_val).intersection(set(pid_test))) == 0
        assert len(set(pid_train).intersection(set(pid_val))) == 0

        artf_frc = [sum(y)/len(y) for y in [ytrain, yval, ytest]]

    return Xtrain, Xval, Xtest, np.array(ytrain), np.array(yval), np.array(ytest)

class ImageReader:
    """
    Class to read and preprocess images from disk.

    :Description: 
        Handles image loading, synthetic augmentation, and preprocessing 
        (resizing, normalization, padding, etc.).

    :Attributes:
        output_size (tuple): Desired output size of the images.
        Synths (dict): Dictionary of synthetic augmentation techniques.
        orient (tio.transforms.ToCanonical): Transformation for RAS+ orientation.
        normalise (tio.transforms.ZNormalization): Transformation for normalization.
        crop_pad (tio.transforms.CropOrPad): Transformation for cropping or padding.
        preprocess (tio.Compose): Composed preprocessing transformations.

    :Methods:
        __init__(output_size: tuple): Initializes the ImageReader with a specified output size.
        _apply_modifications(img_path: str) -> tio.ScalarImage: Applies synthetic modifications to an image based on its path.
        read_image(path: str) -> torch.Tensor: Reads, augments, preprocesses, and returns an image from a given path.
    """

    def __init__(self, output_size: tuple = (192, 256, 256)):
        """
        :param output_size: Desired output size of the images.
        :type output_size: tuple
        """
        self.output_size = output_size

        self.Synths = {
            "RandomAffine": tio.RandomAffine(scales=(1.5, 1.5)),        # zooming in on the images
            "RandomElasticDeformation": tio.RandomElasticDeformation(), # elastic deformation of the images
            "RandomAnisotropy": tio.RandomAnisotropy(),                 # anisotropic deformation of the images
            "RescaleIntensity": tio.RescaleIntensity((0.5, 1.5)),       # rescaling the intensity of the images
            "RandomMotion": tio.RandomMotion(),                         # filling the  𝑘 -space with random rigidly-transformed versions of the original images
            "RandomGhosting": tio.RandomGhosting(),                     # removing every  𝑛 th plane from the k-space
            "RandomSpike": tio.RandomSpike(),                           # signal peak in  𝑘 -space,
            "RandomBiasField": tio.RandomBiasField(),                   # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
            "RandomBlur": tio.RandomBlur(),                             # blurring the images
            "RandomNoise": tio.RandomNoise(),                           # adding noise to the images
            "RandomSwap": tio.RandomSwap(),                             # swapping the phase and magnitude of the images
            "RandomGamma": tio.RandomGamma(),                           # intensity of the images
            # "RandomWrapAround" : self.RandomWrapAround                  # wrapping around the images: Aliasing Artifact
        }
        # pre-processing
        self.orient = tio.transforms.ToCanonical()                  # RAS+ orientation
        self.normalise = tio.transforms.ZNormalization()
        self.crop_pad = tio.transforms.CropOrPad(self.output_size)
        self.preprocess = tio.Compose([self.orient, self.normalise, self.crop_pad])

    def _apply_modifications(self, img_path: str) -> tio.ScalarImage:
        """
        Applies synthetic modifications to an image based on its path.

        :param img_path: Path to the image.
        :type img_path: str
        :return: Modified image.
        :rtype: tio.ScalarImage
        """
        # 1 - Checking the extension on the img_path + storing it as extension_name (i.e the modification to apply)
        extensions = ["CAug", "RandomAffine", 'RandomElasticDeformation',
                      'RandomAnisotropy', 'RescaleIntensity', 'RandomMotion', 'RandomGhosting', 'RandomSpike',
                      'RandomBiasField', 'RandomBlur', 'RandomNoise', 'RandomSwap', 'RandomGamma']  # , 'RandomWrapAround']
        extension_name = None
        for extension in extensions:
            if extension in img_path:
                extension_name = extension
                break

        # 2 - Creating a torchIO image from the path
        if extension_name is None:  # no modification to be applied
            return tio.ScalarImage(img_path)
        stripped_img_path = img_path.replace(f"_{extension_name}.nii", ".nii")
        img = tio.ScalarImage(stripped_img_path)

        # 3 - Defining the modification from extension_name
        if extension_name == "CAug":
            binary_flip = [np.random.choice([0, 1]) for _ in range(3)]
            idx_flip = [index for index, value in enumerate(binary_flip) if value == 1]
            flipping = tio.RandomFlip(axes=idx_flip, flip_probability=1)
            scaling = tio.RandomAffine(scales=(0.1))  # If only one value: U(1-x, 1+x)
            shifting = tio.RandomAffine(translation=(0.1))  # If only one value: U(-x, x)
            rotating = tio.RandomAffine(degrees=(10))  # If only one value: U(-x, x)
            modification = tio.Compose([flipping, scaling, shifting, rotating])

        elif extension_name in ["RandomAffine", 'RandomElasticDeformation', 'RandomAnisotropy', 'RescaleIntensity',
                                'RandomMotion', 'RandomGhosting', 'RandomSpike', 'RandomBiasField', 'RandomBlur',
                                'RandomNoise', 'RandomSwap', 'RandomGamma']:  # , 'RandomWrapAround']:
            modification = self.Synths[extension_name]

        else:
            raise ValueError(f"Path extension (augmentation) {extension_name} not recognised")

        # 4 - Apply the modification on the modified image and return modified version
        return modification(img)

    def read_image(self, path: str) -> torch.Tensor:
        """
        Reads, augments, and preprocesses an image from a given path.

        :param path: Path to the image.
        :type path: str
        :return: Augmented (if pre-specified) and preprocessed image tensor.
        :rtype: torch.Tensor
        """
        img = self._apply_modifications(path)  # introduce artefacts, if specified in path_name
        return self.preprocess(img).tensor  # re-orient, normalise, crop/pad


class DataCrawler:
    """
    The DataCrawler class is responsible for traversing dataset directories to gather and return lists of image paths 
    and their corresponding labels, distinguishing between clean and artefact images.

    Attributes
    ----------
    datadir : str
        Base directory of the dataset.
    datasets : list
        List of dataset names to include.
    image_contrasts : list
        List of image contrast types to include. Can be any number of ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'].
    image_qualities : list
        List of image quality types to include. Must include both ['clean', 'exp_artefacts'].

    Methods
    -------
    __init__(datadir: str, datasets: list, image_contrasts: list, image_qualities: list)
        Initializes the DataCrawler with dataset parameters.
    _check_inputs(datadir: str, datasets: list, image_contrasts: list, image_qualities: list)
        Checks the validity of the input parameters.
    crawl() -> tuple
        Crawls the dataset directories and returns lists of image paths and labels.
    """

    def __init__(self, datadir: str, datasets: list, image_contrasts: list, image_qualities: list):
        """
        Initializes the DataCrawler with dataset parameters.

        :param datadir: Base directory of the dataset.
        :type datadir: str
        :param datasets: List of dataset names to include.
        :type datasets: list
        :param image_contrasts: List of image contrast types to include. Can be any number of ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'].
        :type image_contrasts: list
        :param image_qualities: List of image quality types to include. Must include both ['clean', 'exp_artefacts'].
        :type image_qualities: list
        """
        self._check_inputs(datadir, datasets, image_contrasts, image_qualities)
        self.datadir = datadir
        self.datasets = datasets
        self.image_contrasts = image_contrasts
        self.image_qualities = image_qualities

    def _check_inputs(self, datadir: str, datasets: list, image_contrasts: list, image_qualities: list):
        """
        Checks the validity of the input parameters.

        :param datadir: Base directory of the dataset.
        :type datadir: str
        :param datasets: List of dataset names to include.
        :type datasets: list
        :param image_contrasts: List of image contrast types to include. Can be any number of ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'].
        :type image_contrasts: list
        :param image_qualities: List of image quality types to include. Must be in ['clean', 'exp_artefacts'].
        :type image_qualities: list
        """
        # do paths to all requested types of images exist?
        assert (os.path.isdir(datadir))
        for d in datasets:
            assert (os.path.isdir(os.path.join(datadir, d)))
            for c in image_contrasts:
                assert (c in ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'])
                if os.path.isdir(os.path.join(datadir, c)):
                    for q in image_qualities:
                        assert (q in ['clean', 'exp_artefacts'])
                        assert (os.path.isdir(os.path.join(datadir, c, q)))

    def crawl(self) -> tuple:
        """
        Crawls the dataset directories and returns lists of image paths and labels.

        :return: Tuple containing lists of image paths, patient IDs, and labels.
        :rtype: tuple
        """
        # get all the (non-synthetic) image paths
        clean_img_paths = []
        artefacts_img_paths = []
        for d in self.datasets:
            for c in self.image_contrasts:
                if os.path.isdir(os.path.join(self.datadir, d, c)):
                    for q in self.image_qualities:
                        path = os.path.join(self.datadir, d, c, q)
                        images = os.listdir(path)
                        img_paths = [os.path.join(path, i) for i in images]
                        if q == 'clean':
                            clean_img_paths.extend(img_paths)
                        else:
                            artefacts_img_paths.extend(img_paths)
        # parse out the subject IDs from the paths
        clean_ids = [p.split('sub-')[1].split('_')[0] for p in clean_img_paths]
        artefacts_ids = [p.split('sub-')[1].split('_')[0] for p in artefacts_img_paths]
        # define the appropriate labels
        num_clean = len(clean_img_paths)
        num_artefacts = len(artefacts_img_paths)
        y_true = np.array([0] * num_clean + [1] * num_artefacts)

        return clean_img_paths + artefacts_img_paths, clean_ids + artefacts_ids, y_true


class DataLoader(Sequence):
    """
    DataLoader for an image dataset.

    This class coordinates file reading, augmentation, pre-processing, batching, and passing data to the model.
    It pre-defines augmentations to achieve a target ratio of clean images to artefact-images, if this differs from the underlying dataset.

    :Attributes:
        batch_size (int): Number of images per batch.
        image_shape (tuple): The shape to which images should be cropped/padded.
        num_classes (int): Number of unique classes in the dataset.
        reader (ImageReader): Instance of ImageReader for image preprocessing.
        Clean_img_paths (list): Paths to clean images.
        Artefacts_img_paths (list): Paths to artefact images.
        train_mode (bool): Specifies whether the DataLoader is in training mode.
        target_clean_ratio (float): Desired ratio of clean images to artefact images in training mode.
        target_artef_distro (dict): Distribution of artefact types for generating synthetic artefact images.
        clean_img_paths (list): Paths to all clean images, including synthetic ones.
        artefacts_img_paths (list): Paths to all artefact images, including synthetic ones.
        batches (list): List of batches, each containing image paths.
        labels (list): List of labels corresponding to each batch.

    :Methods:
        __init__(Xpaths, y_true, batch_size, image_shape, train_mode=True, target_clean_ratio=None, artef_distro=None):
            Initializes the DataLoader with the given parameters, checks the validity of inputs, and prepares the dataset for training or evaluation.
        
        _check_inputs(Xpaths, y_true, artef_distro, target_clean_ratio, train_mode):
            Checks the validity of the input parameters for DataLoader.

        on_epoch_end():
            Re-defines new augmentations for the next epoch (if in train mode) and splits image paths into batches.

        __len__():
            Returns the number of batches in the current epoch.

        __getitem__(idx):
            Generates the batch with batch-index `idx`, alongside associated labels.
            Images are read from disk, any pre-specified augmentations and pre-processing is applied and the result passed to the model.

        _def_batches(clean_img_paths, artefacts_img_paths):
            Pre-determines batches at the start of each epoch; images are not read from disk until __getitem__() is called.
            Ensures that the desired ratio of clean images to artefacts (synthetic or not) is reflected in each batch.

        _get_clean_ratio():
            Calculates the current ratio of clean images to total images in the dataset.

        _define_augmentations():
            Pre-defines augmentations to be performed on specific images in order to reach a target ratio of clean images to artefact-images.
    """

    def __init__(self, Xpaths: list, y_true: np.ndarray, batch_size: int, image_shape: tuple,
                 train_mode: bool = True, target_clean_ratio: float = None, artef_distro: dict = None):
        """
        Initializes the DataLoader with the given parameters, checks the validity of inputs, and prepares the dataset for training or evaluation.

        :param Xpaths: List of paths to the image files.
        :type Xpaths: list
        :param y_true: Array of corresponding ground truth labels corresponding to the images.
        :type y_true: numpy.ndarray
        :param batch_size: Number of images per batch.
        :type batch_size: int
        :param image_shape: The shape to which images should be cropped/padded.
        :type image_shape: tuple
        :param train_mode: Specifies whether the DataLoader is in training mode. Defaults to True (in which case synthetic image augmentations can be performed).
        :type train_mode: bool, optional
        :param target_clean_ratio: The desired ratio of clean images to images with artefacts in the dataset. Only relevant in training mode. Defaults to None.
        :type target_clean_ratio: float, optional
        :param artef_distro: Distribution of artefact types for generating synthetic artefact images. Only relevant in training mode. Defaults to None.
        :type artef_distro: dict, optional
        """
        self._check_inputs(Xpaths, y_true, artef_distro, target_clean_ratio, train_mode)

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_classes = len(np.unique(y_true))
        self.reader = ImageReader(output_size=self.image_shape)

        # paths to just the *real* images
        clean_idx, artefact_idx = np.where(y_true == 0)[0], np.where(y_true == 1)[0]
        self.Clean_img_paths = [Xpaths[i] for i in clean_idx]
        self.Artefacts_img_paths = [Xpaths[i] for i in artefact_idx]

        self.train_mode = train_mode
        if self.train_mode:
            self.target_clean_ratio = target_clean_ratio
            self.target_artef_distro = artef_distro

        # paths to all images, including synthetic ones; split into batches
        self.clean_img_paths, self.artefacts_img_paths, self.batches, self.labels = self.on_epoch_end()

    def _check_inputs(self, Xpaths: list, y_true: np.ndarray, artef_distro: dict, target_clean_ratio: float, train_mode: bool):
        """
        Checks the validity of the input parameters for DataLoader.

        :param Xpaths: Paths to the images.
        :type Xpaths: list
        :param y_true: Labels for each image.
        :type y_true: numpy.ndarray
        :param artef_distro: Distribution of artefact types for augmentation.
        :type artef_distro: dict
        :param target_clean_ratio: Target ratio of clean images.
        :type target_clean_ratio: float
        :param train_mode: Whether the loader is in training mode.
        :type train_mode: bool
        """
        assert (len(Xpaths) == len(y_true))
        assert (len(np.unique(y_true)) == 2)
        if train_mode:
            assert (target_clean_ratio >= 0 and target_clean_ratio <= 1)
            # artef_distro defines a categorical probability distribution over (synthetic) artefact types
            assert (isinstance(artef_distro, dict))
            assert (np.allclose(sum(artef_distro.values()), 1))
            for k, v in artef_distro.items():
                assert (v >= 0)
                assert (k in ['RandomAffine', 'RandomElasticDeformation', 'RandomAnisotropy', 'RescaleIntensity',
                              'RandomMotion', 'RandomGhosting', 'RandomSpike', 'RandomBiasField', 'RandomBlur',
                              'RandomNoise', 'RandomSwap', 'RandomGamma'])  # , 'RandomWrapAround'])

    def _get_clean_ratio(self) -> tuple:
        """
        Calculates fraction of clean images (among non-synthetic data).

        :return: Tuple containing the clean ratio, number of clean images, and total images.
        :rtype: tuple
        """
        C = len(self.Clean_img_paths)
        T = C + len(self.Artefacts_img_paths)
        return C / T, C, T

    def _define_augmentations(self) -> tuple:
        """
        Pre-defines augmentations to be performed on specific images in order to reach a target ratio of clean images to artefact-images.
        Desired augmentations are appended to the image path, in a format that can be parsed by `ImageReader`.

        If the required augmentations are synthetic artefacts, these are drawn from a categorical distro over image transforms given to the constructor.
        Synthetic clean images defined by random {flips, shifts, scales, rotations}.

        :return: Tuple containing lists of clean and artefact image paths.
        :rtype: tuple
        """

        def _pick_augment(path: str, aug_type: str = 'clean') -> str:
            """
            Defines which augmentation will be performed for a given image and appends it to the image path.
            Artefacts are randomly drawn from a pre-specified distribution.

            :param path: Path to the image.
            :type path: str
            :param aug_type: Type of augmentation ('clean' or 'artefact').
            :type aug_type: str
            :return: Augmented image path.
            :rtype: str
            """
            if aug_type == 'clean':
                aug = 'CAug'
            elif aug_type == 'artefact':
                augs = list(self.target_artef_distro.keys())
                probs = list(self.target_artef_distro.values())
                aug = np.random.choice(augs, size=1, p=probs)[0]
            else:
                raise ValueError('aug_type must be either "clean" or "artefact"')
            dir, file = os.path.split(path)
            fname, ext = file.split('.', 1)  # just the first dot, so we don't split extensions like .nii.gz
            return os.path.join(dir, fname) + '_' + aug + '.' + ext

        clean_ratio, cleans, total = self._get_clean_ratio()
        clean_img_paths = self.Clean_img_paths
        artefacts_img_paths = self.Artefacts_img_paths

        if clean_ratio < self.target_clean_ratio:
            # oversample clean images with random {flips, shifts, 10% scales, rotations}
            # until desired clean-ratio is reached
            num_imgs_to_aug = int((total * self.target_clean_ratio - cleans) / (1 - self.target_clean_ratio))
            imgs_to_aug = random.choices(clean_img_paths, k=num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='clean') for path in imgs_to_aug]
            clean_img_paths.extend(augmented_paths)
        elif clean_ratio > self.target_clean_ratio:
            # create synthetic artefacts until desired clean-ratio is reached
            # pick augmentations from categorical distro over transform functions of TorchIO
            num_imgs_to_aug = int(cleans / self.target_clean_ratio - total)
            imgs_to_aug = random.choices(clean_img_paths, k=num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='artefact') for path in imgs_to_aug]
            artefacts_img_paths.extend(augmented_paths)

        return clean_img_paths, artefacts_img_paths

    def __len__(self) -> int:
        """
        Returns the number of batches in the current epoch.

        :return: Number of batches.
        :rtype: int
        """
        return len(self.batches)

    def _def_batches(self, clean_img_paths: list, artefacts_img_paths: list) -> tuple:
        """
        Pre-determines batches at start of each epoch; images are not read from disk until __getitem__() is called.

        Ensures that the desired ratio of clean images to artefacts (synthetic or not) is reflected in each batch.

        :param clean_img_paths: List of clean image paths.
        :type clean_img_paths: list
        :param artefacts_img_paths: List of artefact image paths.
        :type artefacts_img_paths: list
        :return: Tuple containing batches and labels.
        :rtype: tuple
        """

        if self.train_mode:
            # 1 - Decide on number of clean and artefact images in each batch with the repartition we target
            num_clean = int(self.batch_size * self.target_clean_ratio)
            num_artefacts = self.batch_size - num_clean

            # 2 - Randomly assign the paths to the batches along with their labels
            nb_batches = len(clean_img_paths + artefacts_img_paths) // self.batch_size
            random.shuffle(self.clean_img_paths)
            random.shuffle(self.artefacts_img_paths)
            batches = [
                self.clean_img_paths[i * num_clean:(i + 1) * num_clean] + self.artefacts_img_paths[
                                                                           i * num_artefacts:(i + 1) * num_artefacts]
                for i in range(nb_batches)
            ]
            labels = [[0] * num_clean + [1] * num_artefacts for _ in range(nb_batches)]

            # 3 - Shuffle batch in batches ALONG with their labels
            for i in range(nb_batches):
                zipped = list(zip(batches[i], labels[i]))
                random.shuffle(zipped)
                batches[i], labels[i] = zip(*zipped)
                batches[i] = list(batches[i])
                labels[i] = list(labels[i])
        else:
            # in this case we don't care about the fraction of cleans in each batch, just chunk the data
            files = clean_img_paths + artefacts_img_paths
            flat_labels = [0] * len(clean_img_paths) + [1] * len(artefacts_img_paths)

            batches, labels = [], []
            while len(files) > 0:
                end_index = min(self.batch_size, len(files))
                batches.append(files[:end_index]);
                labels.append(flat_labels[:end_index])
                files = files[self.batch_size:];
                flat_labels = flat_labels[self.batch_size:]

        return batches, labels

    def __getitem__(self, idx: int) -> tuple:
        """
        Generates the batch with batch-index `idx`, alongside associated labels.
        Images are read from disk, any pre-specified augmentations and pre-processing is applied and the result passed to the model.

        :param idx: Index of the batch.
        :type idx: int
        :return: Tuple containing the batch of images and their one-hot encoded labels.
        :rtype: tuple
        """
        # read in the images, augment with artefacts as necessary, then apply pre-processing
        start_time = time.time()
        batch_images = [self.reader.read_image(path) for path in self.batches[idx]]  # TODO: this takes ~30s for 10-image batch
        print(f"Batch image loading time: {time.time() - start_time} seconds")
        X = torch.stack([img.data.permute(1, 2, 3, 0) for img in batch_images])  # put channel dimension last, then stack along new batch dimension (first)

        # also get appropriate labels
        y_true = self.labels[idx]
        y_one_hot = tf.one_hot(y_true, depth=self.num_classes)

        return X, y_one_hot

    def on_epoch_end(self) -> tuple:
        """
        Re-defines new augmentations for the next epoch (if in train mode).
        Re-defines new batches for next epoch.

        :return: Tuple containing lists of clean and artefact image paths, batches, and labels.
        :rtype: tuple
        """
        # get paths to all images, including synthetic ones, if requested
        if self.train_mode:
            # re-define augmentations to do for next epoch
            self.clean_img_paths, self.artefacts_img_paths = self._define_augmentations()
        else:
            self.clean_img_paths, self.artefacts_img_paths = self.Clean_img_paths, self.Artefacts_img_paths
        # split these new image paths into batches
        self.batches, self.labels = self._def_batches(self.clean_img_paths, self.artefacts_img_paths)

        return self.clean_img_paths, self.artefacts_img_paths, self.batches, self.labels


def plot_train_metrics(history, filename: str):
    """
    Plot training metrics: training loss, accuracy, AUROC, AUPRC.

    :param history: Training history containing metrics.
    :type history: History
    :param filename: Filename to save the plot.
    :type filename: str
    """
    # Plotting 4x4 metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics')

    # Loss subplot
    axs[0, 0].plot(history.history['loss'], label='Train Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Accuracy subplot
    axs[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # AUROC subplot
    axs[1, 0].plot(history.history['auroc'], label='Train AUC')
    axs[1, 0].plot(history.history['val_auroc'], label='Validation AUC')
    axs[1, 0].set_title('AUC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('AUC')
    axs[1, 0].legend()

    # AUPRC subplot
    axs[1, 1].plot(history.history['auprc'], label='Train AP')
    axs[1, 1].plot(history.history['val_auprc'], label='Validation AP')
    axs[1, 1].set_title('AP')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('AP')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(filename)
