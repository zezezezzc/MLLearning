% BP���纯���ƽ�ʵ��
% 1.���ȶ������Һ�����������Ϊ20Hz��Ƶ��Ϊ1Hz
% k = 1; % �趨�����ź�Ƶ��
% p = [0:0.05:4];
% t = cos(k*pi*p) + 3*sin(pi*p);
% plot(p, t, '-'), xlabel('ʱ��'); ylabel('�����ź�');
% % 2.����BP���硣��newff��������ǰ����BP���磬�趨��������Ԫ��ĿΪ10
% % �ֱ�ѡ������Ĵ��ݺ���Ϊ tansig�������Ĵ��ݺ���Ϊ purelin��
% % ѧϰ�㷨Ϊtrainlm��
% net =newff(minmax(p),[10,10,1],{'tansig','tansig','purelin'},'trainlm');
% % 3.�����ɵ�������з��沢��ͼ��ʾ��
% y1 = sim(net,p); plot(p, t, '-', p, y1, '--')
% % 4.ѵ�������������ѵ�����趨ѵ�����Ŀ��Ϊ 1e-5������������Ϊ300��
% % ѧϰ����Ϊ0.05��
% net.trainParam.lr=0.05;
% net.trainParam.epochs=1000;
% net.trainParam.goal=1e-5;
% [net,tr]=train(net,p,t);
% %5.�ٴζ����ɵ�������з��沢��ͼ��ʾ��
% y2 = sim(net,p);
% plot(p, t, '-', p, y2, '--')

% ����
p1=[1.24,1.27;1.36,1.74;1.38,1.64;1.38,1.82;1.38,1.90; 1.40,1.70;1.48,1.82;1.54,1.82;1.56,2.08];
p2=[1.14,1.82;1.18,1.96;1.20,1.86;1.26,2.00 1.28,2.00;1.30,1.96];
p=[p1;p2]';
pr=minmax(p);
goal=[ones(1,9),zeros(1,6);zeros(1,9),ones(1,6)];
plot(p1(:,1),p1(:,2),'h',p2(:,1),p2(:,2),'o')
net=newff(pr,[3,2],{'logsig','logsig'});
net.trainParam.show = 10;
net.trainParam.lr = 0.05;
net.trainParam.goal = 1e-10;
net.trainParam.epochs = 50000;
net = train(net,p,goal);
x=[1.24 1.80;1.28 1.84;1.40 2.04]';
y0=sim(net,p)
y=sim(net,x)
