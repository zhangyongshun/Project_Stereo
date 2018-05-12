# coding=utf-8
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np

class ResidualBlock(nn.Module):

	def __init__(self, in_channels, d1, d2, stride = 1):
		super(ResidualBlock, self).__init__()

		# leading into d1
		self.conv1 = nn.Conv2d(in_channels, d1, 1, stride = stride, bias = False)
		self.bn1 = nn.BatchNorm2d(d1)
		self.relu1 = nn.ReLU(inplace = True)

		# leading into d1-2
		self.conv2 = nn.Conv2d(d1, d1, 3, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(d1)
		self.relu2 = nn.ReLU(inplace = True)

		# leading into d2
		self.conv3 = nn.Conv2d(d1, d2, 1, bias = False)
		self.bn3 = nn.BatchNorm2d(d2)
		self.relu3 = nn.ReLU(inplace = True)


	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)


		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out += x

		out = self.relu3(out)

		return out

class ProjectionBlock(nn.Module):

	def __init__(self, in_channels, d1, d2, stride = 1):
		super(ProjectionBlock, self).__init__()

		# feeding into first d1 block
		self.conv1 = nn.Conv2d(in_channels, d1, 1, stride = stride, bias = False)
		self.bn1 = nn.BatchNorm2d(d1)
		self.relu1 = nn.ReLU(inplace = True)

		# feeding into second d1 block
		self.conv2 = nn.Conv2d(d1, d1, 3, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(d1)
		self.relu2 = nn.ReLU(inplace = True)

		# feeding into first d2 block
		self.conv3 = nn.Conv2d(d1, d2, 1, bias = False)
		self.bn3 = nn.BatchNorm2d(d2)

		# feeding into second d2 block
		self.conv4 = nn.Conv2d(in_channels, d2, 1, stride = stride, bias = False)
		self.bn4 = nn.BatchNorm2d(d2)

		self.relu3 = nn.ReLU(inplace = True)


	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)

		out = self.conv3(out)
		out = self.bn3(out)

		# do residual branch
		residual = x
		residual = self.conv4(residual)
		residual = self.bn4(residual)

		out += residual

		out = self.relu3(out)

		return out

# Fast Up Convolution from the paper, including the interleaving step
class UpConvolution(nn.Module):
	def __init__(self, in_channels, out_channels, batch_size):
		super(UpConvolution, self).__init__()

		self.batch_size = batch_size

		# do 4 convolutions on the same output with different kernels
		self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3))
		self.conv2 = nn.Conv2d(in_channels, out_channels, (2,3))
		self.conv3 = nn.Conv2d(in_channels, out_channels, (3,2))
		self.conv4 = nn.Conv2d(in_channels, out_channels, (2,2))

		self.bn = nn.BatchNorm2d(out_channels)

	def prepare_indices(self, before, row, col, after, dims):

		x0, x1, x2, x3 = np.meshgrid(before, row, col, after)
		x_0 = torch.from_numpy(x0.reshape([-1]))
		x_1 = torch.from_numpy(x1.reshape([-1]))
		x_2 = torch.from_numpy(x2.reshape([-1]))
		x_3 = torch.from_numpy(x3.reshape([-1]))

		linear_indices = x_3 + dims[3] * x_2  + 2 * dims[2] * dims[3] * x_0 * 2 * dims[1] + 2 * dims[2] * dims[3] * x_1
		return linear_indices


	def interleave(self, out1, out2, out3, out4):

		out1 = out1.permute(0, 2, 3, 1)
		out2 = out2.permute(0, 2, 3, 1)
		out3 = out3.permute(0, 2, 3, 1)
		out4 = out4.permute(0, 2, 3, 1)

		dims = out1.size()
		dim1 = dims[1] * 2
		dim2 = dims[2] * 2

		A_row_indices = range(0, dim1, 2)

		A_col_indices = range(0, dim2, 2)
		B_row_indices = range(1, dim1, 2)
		B_col_indices = range(0, dim2, 2)
		C_row_indices = range(0, dim1, 2)
		C_col_indices = range(1, dim2, 2)
		D_row_indices = range(1, dim1, 2)
		D_col_indices = range(1, dim2, 2)

		all_indices_before = range(int(self.batch_size))
		all_indices_after = range(dims[3])

		A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after, dims).cuda()
		B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after, dims).cuda()
		C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after, dims).cuda()
		D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after, dims).cuda()

		A_flat = (out1.permute(1, 0, 2, 3)).contiguous().view(-1)
		B_flat = (out2.permute(1, 0, 2, 3)).contiguous().view(-1)
		C_flat = (out3.permute(1, 0, 2, 3)).contiguous().view(-1)
		D_flat = (out4.permute(1, 0, 2, 3)).contiguous().view(-1)

		size_ = A_linear_indices.size()[0] + B_linear_indices.size()[0]+C_linear_indices.size()[0]+D_linear_indices.size()[0]

		Y_flat = torch.cuda.FloatTensor(size_).zero_()
		#print('a_float', A_linear_indices.squeeze().type())
		Y_flat.scatter_(0, A_linear_indices.squeeze().long(),A_flat.data.float())
		Y_flat.scatter_(0, B_linear_indices.squeeze().long(),B_flat.data.float())
		Y_flat.scatter_(0, C_linear_indices.squeeze().long(),C_flat.data.float())
		Y_flat.scatter_(0, D_linear_indices.squeeze().long(),D_flat.data.float())


		Y = Y_flat.view(-1, dim1, dim2, dims[3])
		Y=Variable(Y.permute(0,3,1,2)).contiguous()

		return Y

	def forward(self, x):

		out1 = self.conv1(nn.functional.pad(x, (1,1,1,1)))
		out2 = self.conv2(nn.functional.pad(x, (1,1,1,0)))
		out3 = self.conv3(nn.functional.pad(x, (1,0,1,1)))
		out4 = self.conv4(nn.functional.pad(x, (1,0,1,0)))
		#print('out1', out1.type())
		out = self.interleave(out1, out2, out3, out4)
		#print('cout1', out.type())

		out = self.bn(out)
		out = out.cuda()

		return out

class UpProjection(nn.Module):

	def __init__(self, in_channels, out_channels, batch_size):
		super(UpProjection, self).__init__()

		self.UpConv1 = UpConvolution(in_channels, out_channels, batch_size)
		self.relu1 = nn.ReLU(inplace = True)
		self.bn = nn.BatchNorm2d(out_channels)

		self.UpConv2 = UpConvolution(in_channels, out_channels, batch_size)

		self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
		self.relu2 = nn.ReLU(inplace = True)

	def forward(self, x):

		out1 = self.UpConv1(x)
		out2 = self.UpConv2(x)

		out1 = self.relu1(out1)
		out1 = self.conv1(out1)
		out1 = self.bn(out1)

		out = out1 + out2
		out = self.relu2(out)

		return out

class FCRN(nn.Module):

	def __init__(self, batch_size = 1):
		super(FCRN, self).__init__()
		self.batch_size = batch_size

		self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 4)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU(inplace = True)
		self.max_pool1 = nn.MaxPool2d(3, stride = 2)

		self.proj1 = ProjectionBlock(64, d1 = 64, d2 = 256, stride = 1)
		self.res1_1 = ResidualBlock(256, d1 = 64, d2 = 256, stride = 1)
		self.res1_2 =  ResidualBlock(256, d1 = 64, d2 = 256, stride = 1)

		self.proj2 = ProjectionBlock(256, d1 = 128, d2 = 512, stride = 2)
		self.res2_1 = ResidualBlock(512, d1 = 128, d2 = 512, stride = 1)
		self.res2_2 = ResidualBlock(512, d1 = 128, d2 = 512, stride = 1)
		self.res2_3 = ResidualBlock(512, d1 = 128, d2 = 512, stride = 1)

		self.proj3 = ProjectionBlock(512, d1 = 256, d2 = 1024, stride = 2)
		self.res3_1 = ResidualBlock(1024, d1 = 256, d2 = 1024)
		self.res3_2 = ResidualBlock(1024, d1 = 256, d2 = 1024)
		self.res3_3 = ResidualBlock(1024, d1 = 256, d2 = 1024)
		self.res3_4 = ResidualBlock(1024, d1 = 256, d2 = 1024)
		self.res3_5 = ResidualBlock(1024, d1 = 256, d2 = 1024)

		self.proj4 = ProjectionBlock(1024, d1 = 512, d2 = 2048, stride = 2)
		self.res4_1 = ResidualBlock(2048, d1 = 512, d2 = 2048)
		self.res4_2 = ResidualBlock(2048, d1 = 512, d2 = 2048)

		self.conv2 = nn.Conv2d(2048, 1024, kernel_size = 1)
		self.bn2 = nn.BatchNorm2d(1024)

		self.UpProj1 = UpProjection(1024, 512, self.batch_size)
		self.UpProj2 = UpProjection(512, 256, self.batch_size)
		self.UpProj3 = UpProjection(256, 128, self.batch_size)
		self.UpProj4 = UpProjection(128, 64, self.batch_size)

		self.conv3 = nn.Conv2d(64, 1, kernel_size = 3, padding = 1)
		self.relu2 = nn.ReLU(inplace = True)


	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.max_pool1(out)

		out = self.proj1(out)
		out = self.res1_1(out)
		out = self.res1_2(out)
		print(out.type())
		out = self.proj2(out)
		out = self.res2_1(out)
		out = self.res2_2(out)
		out = self.res2_3(out)

		out = self.proj3(out)
		out = self.res3_1(out)
		out = self.res3_2(out)
		out = self.res3_3(out)
		out = self.res3_4(out)
		out = self.res3_5(out)

		out = self.proj4(out)
		out = self.res4_1(out)
		out = self.res4_2(out)


		out = self.conv2(out)
		out = self.bn2(out)
		print(out.type())
		out = self.UpProj1(out)
		print(out.type())
		out = self.UpProj2(out)

		out = self.UpProj3(out)

		out = self.UpProj4(out)

		out = self.conv3(out)
		out = self.relu2(out)

		return out
