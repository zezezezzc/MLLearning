clear;
yita1=0.05; 
correct_num=0;%��¼��ȷ������
incorrect_num=0;%��¼��������
test_number=10;%���Լ��У�һ���������֣�9����û��0
test_num=10;%���Լ��У�ÿ�����ֶ��ٸ�����ȥѵ������

load('E:\Matlab_Code\ArtificialNervuseNet\data.mat');
load ('w.mat');%%֮ǰѵ���õ���W�����ˣ�����ֱ�Ӽ��ؽ���
load ('v.mat');

%��¼ʱ��
tic
for number=1:test_number
    numtemp=(number-1)*500;%���ݼ���ÿ��������500��
    for num=1:test_num  %���ƶ�����
        Xtemp=X(num+200+numtemp,:);
        tmp=Xtemp;%������
        tmp=tmp(:);
        %�������������
        x=double(tmp.');      %�������������
        %�õ���������
        y0=x*w;
        %����
        y=1./(1+exp(-y0*yita1));
        %�õ����������
        o0=y*v;
        o=1./(1+exp(-o0*yita1));
        %�����������ʶ�𵽵�����
        [o,index]=sort(o);
        if index(10)==number%�����������λ��Ϊ1 ��Ϊ��ȷ��
            correct_num=correct_num+1;
        else
            incorrect_num=incorrect_num+1;
        % %��ʾ���ɹ������֣���ʾ��Ƚϻ�ʱ��
        %     figure(incorrect_num)
        %     imshow((1-photo_matrix)*255);
        %     title(num2str(number));
        end
    end
end
correct_rate=correct_num/test_number/test_num
toc %��ʱ����
