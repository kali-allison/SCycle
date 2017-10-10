function D = truncateData(longD,burnIn)
D = longD;
D.time = longD.time(burnIn:end);
D.slipVel = longD.slipVel(:,burnIn:end);
D.tauQSPlus = longD.tauQSPlus(:,burnIn:end);
D.tauPlus = longD.tauPlus(:,burnIn:end);
D.slip = longD.slip(:,burnIn:end);
D.maxVel = longD.maxVel(burnIn:end);
% D.surfVel = longD.surfVel(burnIn:end);
D.faultSurfVel = longD.faultSurfVel(burnIn:end);
D.maxTauPlus = longD.maxTauPlus(burnIn:end);

if strcmp(D.dom.problemType,'full')
  D.velMinus = longD.velMinus(:,burnIn:end);
  D.tauQSMinus = longD.tauQSMinus(:,burnIn:end);
  D.uMinus = longD.uMinus(:,burnIn:end);
end

end