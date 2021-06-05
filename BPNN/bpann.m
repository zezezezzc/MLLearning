% ��֪����ѧϰ���н�ʦ��ѧϰ��Ҫ������һ���顯�Լ����ݺ�������ѧϰ   
% BPANN���н�ʦ�ල��ѧϰ(ѵ��������ȷ�����)
clc
clear
close all

nn1=400; %��һ����Ԫ 
nn2=20; %�ڶ�����Ԫ
nn3=10; %��������Ԫ

yita=0.2;         %ѧϰϵ��(�������Ʋ���)
yita1=0.05;     %�������������ż�������Ա�����ֹ���������뺯���ı�����������ȥ�����һ�±仯
cishu=100;      %ѵ������
recog_number=10;        %������ʶ������

���ݿ��ܴ�������
train_num=200;      %ѵ�������У�ÿ�����ֶ�����ͼ�����500��

x=double(zeros(1,nn1));     %�����
yo=double(zeros(1,nn2));        %�м�㣬Ҳ�����ز�
outputo=double(zeros(1,nn3));        %�����
tar_output=double(zeros(1,nn3));        %Ŀ����������������

% ׼��ѵ������
load('E:\Matlab_Code\ArtificialNervuseNet\data.mat');

Xtemp=zeros(1,nn1);     %Ԥ�ȶ�ά
s_record=[0];       %��¼�ܵľ�����
tic%��ʱ

for train_control_num=1:cishu     %ѵ����������  ���  ����ȫ����  �Բ�ͬ��㿪ʼ ��������ֲ���ֵ��
    s=0;
    w=double(rand(nn1,nn2)*2-1);    %��һ��ϵ��   -1~1 ֮��
    cita=double(2*rand(1,nn2)-1);
    delta_w=double(zeros(nn1,nn2)*2-1);
    delta_cita=double(zeros(1,nn2));
    
    v=double(rand(nn2,nn3)*2-1);    %�ڶ���ϵ��
    gaam=double(2*rand(1,nn3)-1);
    delta_v=double(zeros(nn2,nn3));
    delta_gaam=double(zeros(1,nn3));
    
    for number=1:recog_number        %0-9��ѵ��
        numtem=(number-1)*500;       %ѵ������ÿ������500��
        tar_output(number)=1.0;
        for num=1:train_num             %ÿ�����ֶ�����
            Xtemp=X(num+numtem,:);      %ȡÿ��Ԫ��
            x=double(Xtemp);        %�������������
            y0=x*w+cita;
            yo=1./(1+exp(-y0*yita1));%����õ����������
            
            output0=yo*v+gaam;
            outputo=1./(1+exp(-output0*yita1));
            %���չ�ʽ����w��v�ĵ�����ʹ��forѭ���ȽϺķ�ʱ�䣬�����˾���˷�������Ч
            delta_d = (tar_output-outputo).*outputo.*(1-outputo);%������
            delta_e = ((yo.*(1-yo)).').*(v*(delta_d.'));%sum(repmat(delta_d,9,1).*v)%�������ظ�  �л�
                                                                                                   ...�е���չ Ҳ���Լ���չ��Ҳ��չ��%�������   
            delta_w = yita*(x.')*(delta_e.');%wϵ���޸�
            delta_cita=yita*(delta_e.');
            delta_v = yita*(yo.')*(delta_d);
            delta_gaam = gaam.*delta_d;
            %����Ȩֵ
            w=w+delta_w;
            cita=cita+delta_cita;
            v=v+delta_v;
            gaam=gaam+delta_gaam;
            %���������
            s=s+sum((tar_output-outputo).*(tar_output-outputo))/10;%/10;
            s=s/recog_number/train_num   %���ӷֺţ���ʱ������ۿ��������
            train_control_num                     %���ӷֺţ���ʱ������������ۿ�����״̬
            s_record(train_control_num)=s%��¼
        end
    end
end
toc %��ʱ����
plot(1:cishu,s_record);

save('weight.mat','w');
save('cita.mat','cita');
save('veight.mat','v');
save('gaam.mat','gaam');
