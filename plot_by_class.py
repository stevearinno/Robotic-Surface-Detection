import matplotlib.pyplot as plt
import numpy as np

def plot_by_class(data, labels, classes, channel, volume, mean=False):
    """
    Visualizes a channel from the data, each class has their own color
    
    Arguments:
    
        data: data (training data)
        labels: corrensponding labels (classes) for each data point in the channel vector 
        classes: list of all classes (floor types)
        channel: channel number to be plotted
        mean: are we plotting the mean of the class or each sample, default is False
        volume: if mean is False, how many examples will show in plot
    Returns:
        
        Nothing
        
    """
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    
    if not mean:
        plt.figure(figsize=(12,5))
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
        
        for i in range(0, len(classes)):
            y = volume
            print(classes[i])
            for j in range(len(data)):
                if labels[j] == classes[i]:
                    if y > 0:
                        color = colours[i]
                        plt.plot(data[j][channel], label = classes[i], color = color, alpha = 0.7)
                        y = y - 1
            
        plt.legend(bbox_to_anchor=(1, 1))  
        plt.show()
        
    else:
        plt.figure(figsize=(12,5))
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
        
        for i in range(0, len(classes)):
            
            for j in range(len(data)):
                if labels[j] == classes[i]:
                    channeldata = data[j][channel]
                    try:
                        class_channeldata = np.vstack((class_channeldata, channeldata))
                    except:
                        class_channeldata = channeldata
                        
            channel_means = np.zeros(128)    
            
            for k in range(len(channel_means)):
                channel_means[k] = np.mean(class_channeldata[:,k])
            
            try:
                means_by_class = np.vstack((means_by_class, channel_means))
            except:
                means_by_class = channel_means
        
            del class_channeldata
            del channel_means
            del channeldata
        
        for i in range(len(means_by_class)):
            color = colors[i]
            plt.plot(means_by_class[i, :], label = classes[i], alpha = 0.7)
            
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()