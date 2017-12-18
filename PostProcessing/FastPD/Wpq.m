function [result  ] = Wpq(p,q,dp,dq)
%this function for compute (10) formula
p=p(:);
q=q(:);
result=0;
if(dp==dq)
    return;
end

landaC=3.6;
result=  (p( 1 ) - q( 1 )).^2 + ...
         (p( 2 ) - q( 2 )).^2 + ...
         (p( 3 ) - q( 3 )).^2 ;
oghlidosDistanceRGB= sqrt(double(result)          );
expCompute= exp( - oghlidosDistanceRGB/landaC );
result=  max(  [  expCompute(:) ; 0.0003]  );


