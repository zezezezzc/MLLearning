clear;
yita1=0.05; 
correct_num=0;%记录正确的数量
incorrect_num=0;%记录错误数量
test_number=10;%测试集中，一共多少数字，9个，没有0
test_num=10;%测试集中，每个数字多少个，除去训练数据

load('E:\Matlab_Code\ArtificialNervuseNet\data.mat');
load ('w.mat');%%之前训练得到的W保存了，可以直接加载进来
load ('v.mat');

%记录时间
tic
for number=1:test_number
    numtemp=(number-1)*500;%数据集中每个数字有500张
    for num=1:test_num  %控制多少张
        Xtemp=X(num+200+numtemp,:);
        tmp=Xtemp;%行向量
        tmp=tmp(:);
        %计算输入层输入
        x=double(tmp.');      %计算输入层输入
        %得到隐层输入
        y0=x*w;
        %激活
        y=1./(1+exp(-y0*yita1));
        %得到输出层输入
        o0=y*v;
        o=1./(1+exp(-o0*yita1));
        %最大的输出即是识别到的数字
        [o,index]=sort(o);
        if index(10)==number%最大数字所在位置为1 即为正确的
            correct_num=correct_num+1;
        else
            incorrect_num=incorrect_num+1;
        % %显示不成功的数字，显示会比较花时间
        %     figure(incorrect_num)
        %     imshow((1-photo_matrix)*255);
        %     title(num2str(number));
        end
    end
end
correct_rate=correct_num/test_number/test_num
toc %计时结束
