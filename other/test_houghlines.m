
% imgPath = 'paintings/surrealism/max ernst/ConstructedbyMinimaxDadamax.jpg';
% imgPath = 'paintings/surrealism/max ernst/Aquis_Submersus.jpg';
imgPath = 'paintings/Cubism/Jean Mettinger/Jean Metzinger, 1912, Danseuse au cafe, oil on canvas, 146.1 x 114.3 cm, Albright-Knox Art Gallery, Buffalo New York copy.jpg';
I = rgb2gray(imread(imgPath));
rotI = I;%imrotate(I,33,'crop');
BW = edge(rotI,'canny');
[H,T,R] = hough(BW);
figure, imshow(H,[],'XData',T,'YData',R,...
            'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
% hough_threshold = ceil(0.75*max(H(:)));
hough_threshold = 0;
hough_nhoodsize = floor(size(H)/50 /2)*2 * 1 + 1;
disp(hough_threshold)
disp(hough_nhoodsize)
P  = houghpeaks(H,100,'threshold',hough_threshold,'NHoodSize',hough_nhoodsize);
x = T(P(:,2)); y = R(P(:,1));
plot(x,y,'s','color','white');
% Find lines and plot them
lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);
figure, imshow(rotI), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end

% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','blue');
