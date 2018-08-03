function varargout = segmenter_gui(varargin)
% SEGMENTER_GUI MATLAB code for segmenter_gui.fig
%      SEGMENTER_GUI, by itself, creates a new SEGMENTER_GUI or raises the existing
%      singleton*.
%
%      H = SEGMENTER_GUI returns the handle to a new SEGMENTER_GUI or the handle to
%      the existing singleton*.
%
%      SEGMENTER_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEGMENTER_GUI.M with the given input arguments.
%
%      SEGMENTER_GUI('Property','Value',...) creates a new SEGMENTER_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before segmenter_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to segmenter_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help segmenter_gui

% Last Modified by GUIDE v2.5 18-Jun-2018 19:24:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @segmenter_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @segmenter_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before segmenter_gui is made visible.
function segmenter_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to segmenter_gui (see VARARGIN)

% Choose default command line output for segmenter_gui
handles.output = hObject;

handles.rawDir  = uigetdir('C:\','Select directory with raw images');
handles.tifDir  = uigetdir('C:\','Select directory with TIF images');
handles.labelsDir = uigetdir('C:\','Select directory with label images');

handles.rawDir = [handles.rawDir '\'];
handles.tifDir = [handles.tifDir '\'];
handles.labelsDir = [handles.labelsDir '\'];

%handles.rawDir = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\';
%handles.tifDir = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Tif Files\';
%handles.labelsDir = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Labels\';

rawDirInfo = dir([handles.rawDir '*.oct']);

for k = 1:numel(rawDirInfo)
    rawStrings{k} = rawDirInfo(k).name;
end

% Populate the pop up menu
set(handles.popupmenu1,'String', rawStrings');

% Show the first frame of the first pullback
handles.pullbackName = rawDirInfo(1).name(1:(end - 4));
handles.frameNum = 1;

im_raw = imread([handles.rawDir rawDirInfo(1).name], 1);

% Remove guidewire
[guidewire_positions, ~] = remove_guidewire_block(double(im_raw));

% Account for offset
if guidewire_positions(1) < 3
    guidewire_positions(1) = 3;
end
if guidewire_positions(2) < 3
    guidewire_positions(2) = 3;
end
if guidewire_positions(1) > 493
    guidewire_positions(1) = 493;
end
if guidewire_positions(2) > 493
    guidewire_positions(2) = 493;
end

% Detect lumen
im_raw(guidewire_positions(1):guidewire_positions(2), 50:75) = 0;
lumenPixels = lumen_detection_block(double(im_raw), guidewire_positions, false);
            
handles.numALines = size(im_raw, 1);
num_cols = size(im_raw, 2);

set(handles.axes1, 'Position', [27.2 12.2 num_cols handles.numALines]);
set(handles.axes2, 'Position', [1013, 12.2 20 handles.numALines]);

axes(handles.axes1);
imshow(log(double(im_raw) + 1.0), []);hold on;
y_lumen = 1:numel(lumenPixels(:)); 
plot(lumenPixels(:), y_lumen, 'r-', 'LineWidth', 1);

% Show the XY image 
if exist([handles.tifDir handles.pullbackName '.tif'], 'file') ~= 2
    errordlg('XY image not found','File Error');
end

im_xy = imread([handles.tifDir handles.pullbackName '.tif'], 1);
axes(handles.axes3);
imshow(im_xy, []);
set(handles.axes3, 'Position', [1040.2 121 size(im_xy, 2)/2.5 size(im_xy, 1)/2.5]);

% Show the labels for this image
if exist([handles.labelsDir handles.pullbackName '_1_Labels.tif'], 'file') ~= 2
    labels = zeros(size(im_raw, 1), 1);
else
    labels = imread([handles.labelsDir handles.pullbackName '_1_Labels.tif']);
end

handles.labels = labels;
label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes segmenter_gui wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = segmenter_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(handles.figure1)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
handles.pullbackName = contents{get(hObject,'Value')}(1:(end - 4));
handles.frameNum = 1;
set(handles.edit1, 'String', 1);
set(handles.popupmenu2, 'Value', 1); 


axes(handles.axes1);
im_raw = imread([handles.rawDir contents{get(hObject, 'Value')}], 1);

% Remove guidewire
[guidewire_positions, ~] = remove_guidewire_block(double(im_raw));

% Account for offset
if guidewire_positions(1) < 3
    guidewire_positions(1) = 3;
end
if guidewire_positions(2) < 3
    guidewire_positions(2) = 3;
end
if guidewire_positions(1) > 493
    guidewire_positions(1) = 493;
end
if guidewire_positions(2) > 493
    guidewire_positions(2) = 493;
end

% Detect lumen
im_raw(guidewire_positions(1):guidewire_positions(2), 50:75) = 0;
lumenPixels = lumen_detection_block(double(im_raw), guidewire_positions, false);

imshow(log(double(im_raw) + 1.0), []);hold on;
y_lumen = 1:numel(lumenPixels(:)); 
plot(lumenPixels(:), y_lumen, 'r-', 'LineWidth', 1);

if exist([handles.tifDir handles.pullbackName '.tif'], 'file') ~= 2
    errordlg('XY image not found','File Error');
else
    im_xy = imread([handles.tifDir handles.pullbackName '.tif'], 1);
    axes(handles.axes3);
    imshow(im_xy, []);
end

if exist([handles.labelsDir handles.pullbackName '_1_Labels.tif'], 'file') ~= 2
    labels = zeros(size(im_raw, 1), 1);
else
    labels = imread([handles.labelsDir handles.pullbackName '_1_Labels.tif']);
end

handles.labels = labels;
label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.frameNum = str2double(get(hObject, 'String'));

if handles.frameNum < 1
    handles.frameNum = 1;
elseif handles.frameNum > 540
    handles.frameNum = 540;
end

axes(handles.axes1);
im_raw = imread([handles.rawDir handles.pullbackName '.oct'], handles.frameNum);

% Remove guidewire
[guidewire_positions, ~] = remove_guidewire_block(double(im_raw));

% Account for offset
if guidewire_positions(1) < 3
    guidewire_positions(1) = 3;
end
if guidewire_positions(2) < 3
    guidewire_positions(2) = 3;
end
if guidewire_positions(1) > 493
    guidewire_positions(1) = 493;
end
if guidewire_positions(2) > 493
    guidewire_positions(2) = 493;
end

% Detect lumen
im_raw(guidewire_positions(1):guidewire_positions(2), 50:75) = 0;
lumenPixels = lumen_detection_block(double(im_raw), guidewire_positions, false);

imshow(log(double(im_raw) + 1.0), []);hold on;
y_lumen = 1:numel(lumenPixels(:)); 
plot(lumenPixels(:), y_lumen, 'r-', 'LineWidth', 1);

if exist([handles.tifDir handles.pullbackName '.tif'], 'file') ~= 2
    errordlg('XY image not found','File Error');
else
    im_xy = imread([handles.tifDir handles.pullbackName '.tif'], handles.frameNum);
    axes(handles.axes3);
    imshow(im_xy, []);
end

if exist([handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif'], 'file') ~= 2
    labels = zeros(size(im_raw, 1), 1);
else
    labels = imread([handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif']);
end

handles.labels = labels;
label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);


% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
segmenting = contents{get(hObject,'Value')};

[x,y] = ginputax(handles.axes1);

% For every two clicks, make a label
for k = 1:2:numel(x)
    y_start = y(k);
    y_end = y(k+1);

    if y(k) > y(k+1)
        disp('Oops, first click should be above the second click in the image.');
        continue;
    end

    y_start = round(y_start);
    y_end = round(y_end);

    if y_start < 3
        y_start = 1;
    end
    if y_end > handles.numALines  - 3
        y_end = handles.numALines ;
    end

    if strcmp(segmenting, 'Fibrocalcific')
        handles.labels(y_start:y_end) = 1;
    elseif strcmp(segmenting, 'Fibrolipidic')
        handles.labels(y_start:y_end) = 2;
    elseif strcmp(segmenting, 'Other')
        handles.labels(y_start:y_end) = 3;
    end

end

label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);


% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

imwrite(uint16(handles.labels), [handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif']);

% Update handles structure
guidata(hObject, handles);




% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

imwrite(uint16(handles.labels), [handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif']);

if handles.frameNum > 1
    handles.frameNum = handles.frameNum - 1;
end

set(handles.edit1, 'String', handles.frameNum);
set(handles.popupmenu2, 'Value', 1); 

axes(handles.axes1);
im_raw = imread([handles.rawDir handles.pullbackName '.oct'], handles.frameNum);

% Remove guidewire
[guidewire_positions, ~] = remove_guidewire_block(double(im_raw));

% Account for offset
if guidewire_positions(1) < 3
    guidewire_positions(1) = 3;
end
if guidewire_positions(2) < 3
    guidewire_positions(2) = 3;
end
if guidewire_positions(1) > 493
    guidewire_positions(1) = 493;
end
if guidewire_positions(2) > 493
    guidewire_positions(2) = 493;
end

% Detect lumen
im_raw(guidewire_positions(1):guidewire_positions(2), 50:75) = 0;
lumenPixels = lumen_detection_block(double(im_raw), guidewire_positions, false);

imshow(log(double(im_raw) + 1.0), []);hold on;
y_lumen = 1:numel(lumenPixels(:)); 
plot(lumenPixels(:), y_lumen, 'r-', 'LineWidth', 1);

if exist([handles.tifDir handles.pullbackName '.tif'], 'file') ~= 2
    errordlg('XY image not found','File Error');
else
    im_xy = imread([handles.tifDir handles.pullbackName '.tif'], handles.frameNum);
    axes(handles.axes3);
    imshow(im_xy, []);
end

if exist([handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif'], 'file') ~= 2
    labels = zeros(size(im_raw, 1), 1);
else
    labels = imread([handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif']);
end

handles.labels = labels;
label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);


% Update handles structure
guidata(hObject, handles);
    

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

imwrite(uint16(handles.labels), [handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif']);

if handles.frameNum < 540
    handles.frameNum = handles.frameNum + 1;
end

set(handles.edit1, 'String', handles.frameNum);
set(handles.popupmenu2, 'Value', 1); 

axes(handles.axes1);
im_raw = imread([handles.rawDir handles.pullbackName '.oct'], handles.frameNum);

% Remove guidewire
[guidewire_positions, ~] = remove_guidewire_block(double(im_raw));

% Account for offset
if guidewire_positions(1) < 3
    guidewire_positions(1) = 3;
end
if guidewire_positions(2) < 3
    guidewire_positions(2) = 3;
end
if guidewire_positions(1) > 493
    guidewire_positions(1) = 493;
end
if guidewire_positions(2) > 493
    guidewire_positions(2) = 493;
end

% Detect lumen
im_raw(guidewire_positions(1):guidewire_positions(2), 50:75) = 0;
lumenPixels = lumen_detection_block(double(im_raw), guidewire_positions, false);

imshow(log(double(im_raw) + 1.0), []);hold on;
y_lumen = 1:numel(lumenPixels(:)); 
plot(lumenPixels(:), y_lumen, 'r-', 'LineWidth', 1);

if exist([handles.tifDir handles.pullbackName '.tif'], 'file') ~= 2
    errordlg('XY image not found','File Error');
else
    im_xy = imread([handles.tifDir handles.pullbackName '.tif'], handles.frameNum);
    axes(handles.axes3);
    imshow(im_xy, []);
end

if exist([handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif'], 'file') ~= 2
    labels = zeros(size(im_raw, 1), 1);
else
    labels = imread([handles.labelsDir handles.pullbackName '_' num2str(handles.frameNum) '_Labels.tif']);
end

handles.labels = labels;
label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiresume(handles.figure1);

guidata(hObject, handles);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[x,y] = ginputax(handles.axes1);

% For every two clicks, make a label
for k = 1:2:numel(x)
    y_start = y(k);
    y_end = y(k+1);

    if y(k) > y(k+1)
        disp('Oops, first click should be above the second click in the image.');
        continue;
    end

    y_start = round(y_start);
    y_end = round(y_end);

    if y_start < 3
        y_start = 1;
    end
    if y_end > handles.numALines  - 3
        y_end = handles.numALines ;
    end

    handles.labels(y_start:y_end) = 1;


end

label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[x,y] = ginputax(handles.axes1);

% For every two clicks, make a label
for k = 1:2:numel(x)
    y_start = y(k);
    y_end = y(k+1);

    if y(k) > y(k+1)
        disp('Oops, first click should be above the second click in the image.');
        continue;
    end

    y_start = round(y_start);
    y_end = round(y_end);

    if y_start < 3
        y_start = 1;
    end
    if y_end > handles.numALines  - 3
        y_end = handles.numALines ;
    end

    handles.labels(y_start:y_end) = 2;


end

label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[x,y] = ginputax(handles.axes1);

% For every two clicks, make a label
for k = 1:2:numel(x)
    y_start = y(k);
    y_end = y(k+1);

    if y(k) > y(k+1)
        disp('Oops, first click should be above the second click in the image.');
        continue;
    end

    y_start = round(y_start);
    y_end = round(y_end);

    if y_start < 3
        y_start = 1;
    end
    if y_end > handles.numALines  - 3
        y_end = handles.numALines ;
    end
    
    handles.labels(y_start:y_end) = 3;

end

label_color = zeros(size(handles.labels, 1), 3);

label_color(handles.labels == 1, 1) = 1;
label_color(handles.labels == 2, 2) = 1;
label_color(handles.labels == 3, 3) = 1;
label_color = reshape(label_color, size(label_color, 1), 1, 3);

axes(handles.axes2);
imshow(repmat(label_color, 1, 20, 1), []);

% Update handles structure
guidata(hObject, handles);
