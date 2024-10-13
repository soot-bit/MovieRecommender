mport numpy as np
import matplotlib.pyplot as plt

class Plot:
    def frequency_counter(self, plot='user'): 

        """
        Generates data points for plotting frequency distribution for  user/items
        -------------------------------------------------------------------------
            * frequency of users/movies  rating/watched n amout movies/times
        """

        # no of ratings/views per user/movie
        if plot == 'user':
            array = [len(row) for row in self.data_by_user_train] 
            
            # List of ratings 
            ratings = [tup[1] for row in self.data_by_user_train for tup in row]

            #frequencies
            unique_values, counts = np.unique(array, return_counts=True)
            unique_ratings, counts_rating = np.unique(ratings, return_counts = True)
        else:
            array = [len(row) for row in self.data_by_movie_train] 
            
            # List of ratings 
            ratings = [tup[1] for row in self.data_by_movie_train for tup in row]

            #frequencies
            unique_values, counts = np.unique(array, return_counts=True)
            unique_ratings, counts_rating = np.unique(ratings, return_counts = True)

        return unique_values, counts, unique_ratings, counts_rating



    def plot_ratings(self, savefig=True):

        _, _, unique_ratings, counts_ratings = self.frequency_counter()
        
        # Plot histogram 
        plt.bar(unique_ratings, counts_ratings, width=0.4, color='red')
        plt.xlabel('Rating')
        plt.ylabel('Number of Movies')
        plt.title('Movie Ratings Histogram')
        plt.xticks(unique_ratings)
        plt.grid(axis='y', linestyle='--', alpha=0.7)


        # Center 
        plt.xticks(unique_ratings)

       
        if savefig:
            plt.savefig("ratings_plot.svg")
        
        plt.show()


    def plot_scatter(self, savefig=True ):

        """ 
        Generates plot of power laws
        ----------------------------
        Args:
        sparse_user : A user sparse matrix.
        sparse_item : A movie indexed sparse matrix.
        """


        # generate data
        u_x, u_y, _, _ = self.frequency_counter()
        i_x, i_y,_ , _ = self.frequency_counter(plot='item')

    
        # freq scatter graph
        fig, ax = plt.subplots()
        ax.scatter(u_x,u_y, label = "user", marker = "o", s=2)
        ax.scatter(i_x,i_y ,label = "item", marker = "^", s=2)

        # axis labelssavefig = False
        ax.set_title("Power law 25M data" , fontsize=16)
        ax.legend( fontsize=10 )
        ax.set_xlabel("degree", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        if savefig:
            plt.savefig('Powerlaw.svg', bbox_inches='tight')



        # Set the default figure size
        ax.grid(True)
        plt.rcParams['font.size'] = 12 # fon ot size
        ax.set_yscale("log")
        ax.set_xscale("log")
        if savefig:
            plt.savefig('Powerlaw_logscale.svg')  
        
        plt.show()




def plot_loss(cost, cost_test, save_fig=True, filename="loss.svg"):

    """
    Plots the loss function or RMSE
    ------------------------------

    Args:
        cost: A list of the training losses.
        save_fig: Whether to save the plot to a file.
    """

    plt.rcParams["font.size"] = 12 # default font
    
    x_val = np.arange(0, len(cost), 1)

    fig, ax = plt.subplots()
    ax.plot(x_val, cost, "r:", linewidth = 2.5, label="Train")
    ax.legend(loc="upper left", fontsize=12)

    # labels and title
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs iterarations", fontsize=15)
    ax.grid(True)

    # save fig?
    if save_fig:
        plt.savefig(filename, bbox_inches="tight")


    plt.show()


def plot_rmse(cost, cost_test, save_fig=True, filename="RMSE.svg"):

    """ Plots RMSEs """


    plt.rcParams["font.size"] = 12
    

    fig, ax = plt.subplots()


    ax.plot(np.arange(0, len(cost), 1), cost, "r:", linewidth = 2.5, label="Train")
    ax.plot(np.arange(0, len(cost_test), 1), cost_test, "b:", linewidth = 2.5, label="Test")
    ax.legend(loc="upper left", fontsize=12)

    # labels and title
    ax.set_xlabel("Iterations")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs iterarations", fontsize=15)
    ax.grid(True)

    # ?
    if save_fig:
        plt.savefig(filename, bbox_inches="tight")


    plt.show()
