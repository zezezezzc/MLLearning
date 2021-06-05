% BP网络函数逼近实例
% 1.首先定义正弦函数，采样率为20Hz，频率为1Hz
% k = 1; % 设定正弦信号频率
% p = [0:0.05:4];
% t = cos(k*pi*p) + 3*sin(pi*p);
% plot(p, t, '-'), xlabel('时间'); ylabel('输入信号');
% % 2.生成BP网络。用newff函数生成前向型BP网络，设定隐层中神经元数目为10
% % 分别选择隐层的传递函数为 tansig，输出层的传递函数为 purelin，
% % 学习算法为trainlm。
% net =newff(minmax(p),[10,10,1],{'tansig','tansig','purelin'},'trainlm');
% % 3.对生成的网络进行仿真并做图显示。
% y1 = sim(net,p); plot(p, t, '-', p, y1, '--')
% % 4.训练。对网络进行训练，设定训练误差目标为 1e-5，最大迭代次数为300，
% % 学习速率为0.05。
% net.trainParam.lr=0.05;
% net.trainParam.epochs=1000;
% net.trainParam.goal=1e-5;
% [net,tr]=train(net,p,t);
% %5.再次对生成的网络进行仿真并做图显示。
% y2 = sim(net,p);
% plot(p, t, '-', p, y2, '--')

% 例程
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
