function D = truncateData(longD,burnIn)
D = longD;
D.time = longD.time(burnIn:end);
D.vel = longD.vel(:,burnIn:end);
D.tau = longD.tau(:,burnIn:end);
D.faultDisp = longD.faultDisp(:,burnIn:end);
D.maxVel = longD.maxVel(burnIn:end);
D.surfVel = longD.surfVel(burnIn:end);
D.maxTau = longD.maxTau(burnIn:end);

end