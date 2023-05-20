import mxnet as mx
from network.layer import Reconstruction2DSmooth
import PIL.Image as Image
import numpy as np

class Test:
    def __init__(self):
        self.reconstruction = Reconstruction2DSmooth(3)
        self.reconstruction.hybridize()

    def composite_flow(self, flow1, flow2):
        """composite two flow fields

        Args:
            flow1: (B, 2, H, W)
            flow2: (B, 2, H, W)

        Returns:
            flow: (B, 2, H, W)
        """
        # if not nd array
        if not isinstance(flow1, mx.nd.NDArray):
            flow1 = mx.nd.array(flow1).as_in_context(mx.gpu())
        if not isinstance(flow2, mx.nd.NDArray):
            flow2 = mx.nd.array(flow2).as_in_context(mx.gpu())
        flow = flow2 + self.reconstruction(flow1, flow2)
        return flow

    def test(self, flow, inv_flow) -> np.array:
        """
        Params:
            flow: (1, 2, H, W)"""
        # turn flow and inv_flow to mx.nd in gpu
        flow = mx.nd.array(flow).as_in_context(mx.gpu())
        inv_flow = mx.nd.array(inv_flow).as_in_context(mx.gpu())
        c_flow = self.composite_flow(flow, inv_flow)
        # turn flow as a 2d img
        flow_img = mx.nd.sqrt(mx.nd.sum(c_flow ** 2, axis=1))
        # turn flow_img to numpy
        flow_img = flow_img.asnumpy()
        return flow_img
