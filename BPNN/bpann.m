% 感知器的学习是有教师的学习，要给他‘一本书’以及传递函数进行学习   
% BPANN是有教师监督的学习(训练数据有确定输出)
clc
clear
close all

nn1=400; %第一层神经元 
nn2=20; %第二层神经元
nn3=10; %第三层神经元

yita=0.2;         %学习系数(概念类似步长)
yita1=0.05;     %比例参数，缩放激活函数的自变量防止输入过大进入函数的饱和区，可以去掉体会一下变化
cishu=100;      %训练次数
recog_number=10;        %共多少识别数字

数据可能存在问题
train_num=200;      %训练样本中，每种数字多少张图，最大500张

x=double(zeros(1,nn1));     %输入层
yo=double(zeros(1,nn2));        %中间层，也是隐藏层
outputo=double(zeros(1,nn3));        %输出层
tar_output=double(zeros(1,nn3));        %目标输出，即理想输出

% 准备训练数据
load('E:\Matlab_Code\ArtificialNervuseNet\data.mat');

Xtemp=zeros(1,nn1);     %预先定维
s_record=[0];       %记录总的均方差
tic%计时

for train_control_num=1:cishu     %训练次数控制  大概  次完全够了  以不同起点开始 尽量避免局部极值点
    s=0;
    w=double(rand(nn1,nn2)*2-1);    %第一层系数   -1~1 之间
    cita=double(2*rand(1,nn2)-1);
    delta_w=double(zeros(nn1,nn2)*2-1);
    delta_cita=double(zeros(1,nn2));
    
    v=double(rand(nn2,nn3)*2-1);    %第二层系数
    gaam=double(2*rand(1,nn3)-1);
    delta_v=double(zeros(nn2,nn3));
    delta_gaam=double(zeros(1,nn3));
    
    for number=1:recog_number        %0-9的训练
        numtem=(number-1)*500;       %训练库中每种数字500张
        tar_output(number)=1.0;
        for num=1:train_num             %每个数字多少张
            Xtemp=X(num+numtem,:);      %取每行元素
            x=double(Xtemp);        %计算输入层输入
            y0=x*w+cita;
            yo=1./(1+exp(-y0*yita1));%激活，得到输出层输入
            
            output0=yo*v+gaam;
            outputo=1./(1+exp(-output0*yita1));
            %按照公式计算w和v的调整，使用for循环比较耗费时间，采用了矩阵乘法，更高效
            delta_d = (tar_output-outputo).*outputo.*(1-outputo);%输出误差
            delta_e = ((yo.*(1-yo)).').*(v*(delta_d.'));%sum(repmat(delta_d,9,1).*v)%将矩阵重复  行或
                                                                                                   ...列的拓展 也可以既扩展行也扩展列%隐层误差   
            delta_w = yita*(x.')*(delta_e.');%w系数修改
            delta_cita=yita*(delta_e.');
            delta_v = yita*(yo.')*(delta_d);
            delta_gaam = gaam.*delta_d;
            %更新权值
            w=w+delta_w;
            cita=cita+delta_cita;
            v=v+delta_v;
            gaam=gaam+delta_gaam;
            %计算均方差
            s=s+sum((tar_output-outputo).*(tar_output-outputo))/10;%/10;
            s=s/recog_number/train_num   %不加分号，随时输出误差观看收敛情况
            train_control_num                     %不加分号，随时输出迭代次数观看运行状态
            s_record(train_control_num)=s%记录
        end
    end
end
toc %计时结束
plot(1:cishu,s_record);

save('weight.mat','w');
save('cita.mat','cita');
save('veight.mat','v');
save('gaam.mat','gaam');
