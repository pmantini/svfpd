from yolov4.tf import YOLOv4
import numpy as np
import tensorflow as tf

class myyolov4(YOLOv4):

    @tf.function
    def _predict(self, x, batch_size):
        # s_pred, m_pred, l_pred
        # x_pred == Dim(1, output_size, output_size, anchors, (bbox))

        candidates = self.model(x, training=False)

        _candidates = []
        for k in range(batch_size):
            this_image = (tf.expand_dims(candidates[0][k], axis=0), tf.expand_dims(candidates[1][k], axis=0), tf.expand_dims(candidates[2][k], axis=0))
            this_candidates = []
            for candidate in this_image:
                grid_size = candidate.shape[1:3]
                this_candidates.append(
                    tf.reshape(
                        candidate[0], shape=(1, grid_size[0] * grid_size[1] * 3, -1)
                    )
                )

            _candidates += [tf.concat(this_candidates, axis=1)]

        return _candidates

    def predict(
            self,
            frames: np.ndarray,
            iou_threshold: float = 0.3,
            score_threshold: float = 0.25,
    ):
        """
        Predict one frame

        @param frame: Dim(height, width, channels)

        @return pred_bboxes == Dim(-1, (x, y, w, h, class_id, probability))
        """
        # image_data == Dim(1, input_size[1], input_size[0], channels)

        image_data = frames

        candidates = self._predict(image_data, image_data.shape[0])
        pred_bboxes_b = []
        for k in range(image_data.shape[0]):

            pred_bboxes = self.candidates_to_pred_bboxes(
                candidates[k][0].numpy(),
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            pred_bboxes_b += [self.fit_pred_bboxes_to_original(pred_bboxes, image_data[0].shape)]

        return pred_bboxes_b
