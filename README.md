# EB-GS-DualNet_2D-tracking-of-robotic-surgery-instruments

Authors: 

Weian Wang, Fahd Qureshi

Supervisors: 

Agostino Stilli, Beatrice Van Amsterdam, Emanuele Colleoni, Frans Chadebecq

Abstract: 

Robotically-assisted minimally invasive surgery (RAMIS) has became a popularly applied technique and gold standard for a wide field of abdominal surgery. Although this technique inherits both advantages of minimally invasive surgery and telesurgery, it requires more accurate and stable surgical instrument detection and tracking capabilities to provide safe tool-tissue interaction. Despite recent advances of learning-based approaches, the real-time tracking and segmentation of surgical tools remain challenging problems. To address these problems, we propose to utilize event-based channel. Event-based cameras embed imaging-sensors responding to local change in brightness which allow them to capture movement and generate videos with high dynamic range and temporal resolution. In this paper, we present a novel architecture, called EB-GS DualNet, as a real-time learning-based technique to automatically track articulated instruments in RAMIS (endoscopic images) by using combination of grayscale and event-based images. This network has proven to have better accuracy than popular models such as Transformer and Xception on surgical tool segmentation task.

Code Description:

data -> all data for training and testing (empty due to large size, please contact author to get dataset)

predictions -> the predicted result from each model and each dataset

TransUNet-single -> all files to run TransUNet model

Xception -> all files to run Xception model