from typing import List
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, glob, shutil
import contextlib
from PIL import Image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def getEventsFilesSorted(directory:str):
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter files that start with 'events.out.tfevents'
    filtered_files = [directory+'/'+f for f in files if f.startswith('events.out.tfevents')]
    
    # Sort the filtered list alphabetically
    sorted_files = sorted(filtered_files)
    
    return sorted_files

def extractImagesFromEvents(tensorboard_directory: str, image_tag:str):
    """ Generator that returns images saved to Tensorboard summaries. """
    for event_file in getEventsFilesSorted(tensorboard_directory):
        serialized_examples = tf.data.TFRecordDataset(event_file)
        image_tags = [image_tag] #stopgap fix
        try:
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for v in event.summary.value:
                    if v.tag in image_tags:
                        if v.HasField('image'):  # event for images using tensor field
                            s = v.image.encoded_image_string
                            
                            tf_img = tf.image.decode_image(s)  # [H, W, C]
                            np_img = tf_img.numpy()

                            yield np_img
        except tf.errors.OutOfRangeError:
            print(f"End of sequence reached for file: {event_file}")
            
def saveEventImages(tensorboard_directory:str, tag:str, output_folder_name:str, plot_title:str = ''):
    """ Saves images extracted from Tensorboard summaries with an embedded title. 
        For use in GIF generation.
    """
    os.makedirs(output_folder_name, exist_ok=True)
    step = 0
    image_extractor = extractImagesFromEvents(tensorboard_directory, tag)
    for image in image_extractor:
        plt.imshow(image)
        if plot_title!='':
            plt.title(plot_title+', Epoch '+str(step+1))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_folder_name+'/'+tag.split('/')[-1]+'_epoch_'+f'{(step+1):03}'+'.png')
        step += 1
    plt.close()
    print("Successfully extracted all images.")

def saveImagesToGIF(filepaths_in, filepath_out, ms_per_img=75, delete_imgs_after_use=False):
    """ Convert images in a given directory to a GIF. Optionally,
        delete the files after use. 
        
        This was adapted from a StackOverflow post here:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    """
    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

            # lazily load images
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(filepaths_in)))
            num_frames = len(glob.glob(filepaths_in))
            # extract  first image from iterator
            img = next(imgs)

            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=filepath_out, format='GIF', append_images=imgs,
                    save_all=True, duration=num_frames*ms_per_img, loop=0)

    if delete_imgs_after_use:
        shutil.rmtree(os.path.dirname(filepaths_in))

def combinePredsWithGT(gt_fname, pred_fname, out_dir, figsize, plot_title:str=''):
    os.makedirs(out_dir, exist_ok=True)
    ground_truth_imgs = sorted(glob.glob(gt_fname))
    pred_imgs = sorted(glob.glob(pred_fname))

    cmap = np.array([
        [220, 20, 60],   # person
        [  0,  0, 142],  # car
        [  0,  0,  70],  # truck
        [  0,  60, 100], # bus
        [  0,  80, 100], # train
        [  0,  0, 230],  # motorcycle
        [119, 11,  32],  # bicycle
        [156, 14, 168],  # motorcyclist
        [177, 16,  48]   # bicyclist
    ], dtype=np.uint8)
    for iter, (gt, pred) in enumerate(zip(ground_truth_imgs, pred_imgs)):
        gt_img = plt.imread(gt)
        pred_img = plt.imread(pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(gt_img)
        ax1.axis('off')
        ax1.set_title('Ground Truth', y=0.9)
        ax2.imshow(pred_img)
        ax2.axis('off')
        ax2.set_title('Prediction', y=0.9)
        
        # Create legend for box colors
        legend_labels = ['Person', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle', 'Motorcyclist', 'Bicyclist']
        legend_patches = [Rectangle((0, 0), 1, 1, color=cmap[i] / 255., alpha=0.5) for i in range(len(legend_labels))]
        ax1.legend(legend_patches, legend_labels, loc='right', fontsize='x-small')
        if plot_title!='':
            fig.suptitle(plot_title+f', Epoch {iter+1}',fontweight='bold')
        # Adjust layout
        plt.tight_layout()
        # plt.subplots_adjust(top=0.85, right=0.85, wspace=0.1, hspace=0.2)
        plt.savefig(out_dir+'/combined_'+str(iter)+'.png')

        plt.close()

if __name__ == "__main__":
    saveEventImages('D_train_experiment_0_with_3_trainable_layers', 'val/pred', '../D_train_saved_images_test', 'Depth FasterRCNN')
    saveImagesToGIF('../D_train_saved_images_test/*.png', 'test.gif', 75, True)