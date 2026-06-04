
%% Export Simulink/MATLAB data for the Python Rerun visualizer

tvec = out.tout;
q    = squeeze(out.q);
qd   = squeeze(out.qd);
pb1  = squeeze(out.b1);
pb2  = squeeze(out.b2);
b1d  = squeeze(out.b1d);
b2d  = squeeze(out.b2d);

l1 = config.l1;
switching_instants = config.switching_instants;
p_des = config.p_des;

activeIntervals = [
    0, 4;
    4, 7;
    7, 9;
    9, 13;
    13, 18;
    18, 22;
];

activeVars = {
    'q3,pb1x,pb1y';
    'q2,pb2x,pb2y';
    'q3,pb1x,pb1y';
    'q3,pb2x,pb2y';
    'q1,pb2x,pb2y';
    'q1,q2,q3';
};

save('datarecord/simulink_3r_export.mat', ...
    'tvec', 'q', 'qd', 'pb1', 'pb2', 'b1d', 'b2d', ...
    'l1', 'switching_instants', 'p_des', ...
    'activeIntervals', 'activeVars');

disp('Saved simulink_3r_export.mat for the Python Rerun visualizer.');
