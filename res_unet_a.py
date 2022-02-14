"""
Implementation of ResUNet-a (with some modifications): https://arxiv.org/abs/1904.00592

Author: Gerben van Veenendaal (gerben@aidence.com)
"""

import torch
import torch.nn.functional as F


class Initialization(torch.nn.Module):
    def __init__(self, output_num_filters):
        super().__init__()

        self.convolution = torch.nn.Conv2d(1, output_num_filters, (1, 1), bias=True)

    def forward(self, input):
        start_value, end_value = -1000, 300

        input = torch.clip(
            (input - start_value) * (2.0 / (end_value - start_value)) - 1.0, -1.0, 1.0
        )

        return self.convolution(input)


class ResNet(torch.nn.Module):
    def __init__(self, num_filters, dilations):
        super().__init__()

        self.common_instance_normalization = torch.nn.InstanceNorm2d(
            num_filters, affine=True
        )
        self.first_convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    num_filters,
                    num_filters // 2,
                    (3, 3),
                    dilation=dilation,
                    padding=(dilation, dilation),
                    bias=True,
                )
                for dilation in dilations
            ]
        )
        self.instance_normalizations = torch.nn.ModuleList(
            [torch.nn.InstanceNorm2d(num_filters // 2, affine=True) for _ in dilations]
        )
        self.second_convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    num_filters // 2,
                    num_filters,
                    (3, 3),
                    dilation=dilation,
                    padding=(dilation, dilation),
                    bias=False,
                )
                for dilation in dilations
            ]
        )

    def forward(self, input):
        common_output = self.common_instance_normalization(input)
        common_output = F.relu(common_output)

        sum_outputs = [input]
        for first_convolution, instance_normalization, second_convolution in zip(
            self.first_convolutions,
            self.instance_normalizations,
            self.second_convolutions,
        ):
            sum_output = first_convolution(common_output)
            sum_output = instance_normalization(sum_output)
            sum_output = F.relu(sum_output)
            sum_output = second_convolution(sum_output)

            sum_outputs.append(sum_output)

        summed_output = torch.sum(torch.stack(sum_outputs, dim=0), dim=0)

        return summed_output


class Upscaling(torch.nn.Module):
    def __init__(self, input_num_filters, output_num_filters):
        super().__init__()

        self.convolution = torch.nn.ConvTranspose2d(
            input_num_filters, output_num_filters, (2, 2), stride=(2, 2), bias=False
        )

    def forward(self, input):
        return self.convolution(input)


class Downscaling(torch.nn.Module):
    def __init__(self, input_num_filters, output_num_filters):
        super().__init__()

        self.convolution = torch.nn.Conv2d(
            input_num_filters, output_num_filters, (2, 2), stride=(2, 2), bias=False
        )

    def forward(self, input):
        return self.convolution(input)


class Combining(torch.nn.Module):
    def __init__(self, first_num_filters, second_num_filters, output_num_filters):
        super().__init__()

        self.instance_normalization = torch.nn.InstanceNorm2d(
            first_num_filters + second_num_filters, affine=True
        )
        self.convolution = torch.nn.Conv2d(
            first_num_filters + second_num_filters,
            output_num_filters,
            (1, 1),
            bias=False,
        )

    def forward(self, first, second):
        output = torch.cat([first, second], dim=1)

        output = self.instance_normalization(output)
        output = self.convolution(output)

        return output


class ResUNetA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._num_filters = [
            16,  # Input size: 256.
            16,  # Input size: 128.
            16,  # Input size: 64.
            16,  # Input size: 32.
            16,  # Input size: 16.
        ]
        self._dilations = [
            [1, 9, 17],  # Input size: 256.
            [1, 5, 9],  # Input size: 128.
            [1, 3, 5],  # Input size: 64.
            [1, 2, 3],  # Input size: 32.
            [1],  # Input size: 16.
        ]
        self._num_stages = len(self._num_filters)

        self.initialization = Initialization(self._num_filters[0])
        self.down_res_nets = torch.nn.ModuleList(
            [
                ResNet(stage_num_filters, stage_dilations)
                for stage_num_filters, stage_dilations in zip(
                    self._num_filters, self._dilations
                )
            ]
        )
        self.up_res_nets = torch.nn.ModuleList(
            [
                ResNet(stage_num_filters, stage_dilations)
                for stage_num_filters, stage_dilations in zip(
                    self._num_filters[:-1], self._dilations[:-1]
                )
            ]
        )
        self.downscalings = torch.nn.ModuleList(
            [
                Downscaling(
                    self._num_filters[stage_index], self._num_filters[stage_index + 1]
                )
                for stage_index in range(self._num_stages - 1)
            ]
        )
        self.upscalings = torch.nn.ModuleList(
            [
                Upscaling(
                    self._num_filters[stage_index + 1], self._num_filters[stage_index]
                )
                for stage_index in range(self._num_stages - 1)
            ]
        )
        self.combinings = torch.nn.ModuleList(
            [
                Combining(stage_num_filters, stage_num_filters, stage_num_filters)
                for stage_num_filters in self._num_filters
            ]
        )

        self.final_combining = Combining(
            self._num_filters[0], self._num_filters[0], self._num_filters[0]
        )

        self.final_convolution = torch.nn.Conv2d(
            self._num_filters[0], 1, (1, 1), bias=True
        )

    def forward(self, input):
        initialization_output = self.initialization(input)

        combine_inputs = []
        output = initialization_output
        for stage_index in range(self._num_stages - 1):
            output = self.down_res_nets[stage_index](output)
            combine_inputs.append(output)
            output = self.downscalings[stage_index](output)

        output = self.down_res_nets[self._num_stages - 1](output)

        for stage_index in reversed(range(self._num_stages - 1)):
            output = self.upscalings[stage_index](output)
            output = self.combinings[stage_index](output, combine_inputs[stage_index])
            output = self.up_res_nets[stage_index](output)

        output = self.final_combining(output, initialization_output)
        output = self.final_convolution(output)

        return output
