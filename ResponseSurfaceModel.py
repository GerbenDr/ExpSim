# Import necessary packages
import numpy as np
import pandas as pd
import constants as c
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import norm

#TODO: update keys if relevant
keys_to_model = ['CL', 'CD', 'CMpitch']

def unpack_RSM_data(dataframe):
    AOA = dataframe['AoA'].to_numpy()
    DELTA_E = dataframe['delta_e'].to_numpy()
    J = 0.5 * (dataframe['J_M1'] +  dataframe['J_M2']).to_numpy()
    data = np.vstack((
            np.ones(AOA.shape),
            AOA,
            J,
            DELTA_E,
            AOA * J,
            J * DELTA_E,
            AOA * DELTA_E,
            AOA * J * DELTA_E,
            AOA ** 2,
            AOA**2 * DELTA_E,
            AOA**3,
            J**2,
            J**3,
            J**2 * DELTA_E,
            J**2 * AOA,
            J * AOA **2,
            J**2 * AOA **2,
            J**2 * AOA * DELTA_E,
            J * AOA**2 * DELTA_E,
            J**2 * AOA**2 * DELTA_E,
        ))
    
    return data

class ResponseSurfaceModel:
    def __init__(self, dataframe: pd.DataFrame, validation_dataframe:pd.DataFrame = None):
        self.ground_truth = {key : dataframe[key] for key in keys_to_model}
        self.coefficients = {key: np.zeros(20) for key in keys_to_model}

        self.data = unpack_RSM_data(dataframe)
        self.dataframe = dataframe
        self.fit()

        self.validation_dataframe = validation_dataframe
        if self.validation_dataframe is not None:
            result, residuals, loss, deltas = self.predict(validation_dataframe)
            self.validation_data = unpack_RSM_data(validation_dataframe)

            self.validation_res = residuals
            self.validation_loss = loss
            self.validation_deltas = deltas
            self.validation_ground_truth = {key : validation_dataframe[key] for key in keys_to_model}


    def _evaluate(self,coefficients, data=None):
        if data is None:
            data = self.data
        return np.dot(coefficients, data)
    
    def _evaluate_from_AJD(self, coefficients, AOA, J, DELTA_E):
        data = np.stack((
            np.ones(AOA.shape),
            AOA,
            J,
            DELTA_E,
            AOA * J,
            J * DELTA_E,
            AOA * DELTA_E,
            AOA * J * DELTA_E,
            AOA ** 2,
            AOA**2 * DELTA_E,
            AOA**3,
            J**2,
            J**3,
            J**2 * DELTA_E,
            J**2 * AOA,
            J * AOA **2,
            J**2 * AOA **2,
            J**2 * AOA * DELTA_E,
            J * AOA**2 * DELTA_E,
            J**2 * AOA**2 * DELTA_E,
        ), axis=-1)
         
        return np.tensordot(coefficients, data, axes=(0, -1))


    def fit(self, ddof = 1):
        result = {}
        residuals = {}
        deltas = {}
        for key, var in self.ground_truth.items():
            res = minimize(lambda coefficients: np.sum((self._evaluate(coefficients) - var)**2), self.coefficients[key])  # minimize sum of residuals
            result[key] = res.x
            residuals[key] = res.fun
            deltas[key] = self._evaluate(res.x) - var

        self.coefficients = result
        self.residuals = residuals
        training_loss = {key:value / (self.data.shape[1] - ddof) for key, value in residuals.items()} # LOSS = MEAN RESIDUAL = VARIANCE OF DIFFERENCES FROM RSM
        self.training_loss = training_loss

        self.training_deltas = deltas
        return result, residuals, training_loss, deltas

    def predict(self, dataframe: pd.DataFrame, ddof = 1):

        data_ext = unpack_RSM_data(dataframe)
        result = {}
        residuals = {}
        deltas = {}
        for key, var in self.ground_truth.items():
            result[key] = self._evaluate(self.coefficients[key], data_ext)
            residuals[key] = np.sum((result[key] - dataframe[key])**2) 
            deltas[key] = result[key] - dataframe[key]

        loss = {key:value / (data_ext.shape[1] - ddof) for key, value in residuals.items()} # LOSS = MEAN RESIDUAL = VARIANCE OF DIFFERENCES FROM RSM

        # TODO: Better output?
        return result, residuals, loss, deltas
    
    @property
    def coefficient_covariance(self):
        return np.linalg.inv(
            self.data @ self.data.T
        )

    @property
    def prediction_covariance(self):
        return self.data.T @ np.linalg.inv(
            self.data @ self.data.T
        ) @ self.data
    

    def print_hypothesis_test_results(self, alpha=0.05, beta=0.01, K = 2 * np.sqrt(2)):

        if self.validation_dataframe is None:
            print('Warning: attempting to perform hypothesis test, but no validation data was provided')
            return

        for key in keys_to_model:
            std_tr = np.sqrt(self.training_loss[key])
            std_val = np.sqrt(self.validation_loss[key])
            mean_val = np.abs(self.validation_deltas[key].mean())

            N = len(self.validation_deltas[key])

            tolerance = K * std_tr

            z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed for alpha - deviation either direction is significant

            z_alpha_actual = (mean_val * np.sqrt(N)) / std_tr

            z_beta = norm.ppf(1 - beta)  # 1-tailed for beta - only consider deviation in the direction of valid model

            z_beta_actual = ((tolerance - mean_val) * np.sqrt(N)) / std_tr

            print('\n')
            print('Hypothesis test results for {}:'.format(key))
            # print('Type 1 error probability: {:.8f}%'.format(100 * 2 * (1 - norm.cdf(z_alpha_actual))))
            # print('Type 2 error probability: {:.8f}%'.format(100 * (1-norm.cdf(z_beta_actual))))
            print('z_alpha_actual: {:.4f}, z_alpha_bound:{:.4f}'.format(z_alpha_actual, z_alpha))
            print('Type 1 acceptable? {}'.format(z_alpha_actual < z_alpha))
            print('z_beta_actual: {:.4f}, z_beta_bound:{:.4f}'.format(z_beta_actual, z_beta))
            print('Type 2 acceptable? {}'.format(z_beta_actual > z_beta))
            print(('Model accepted' if all([z_alpha_actual < z_alpha, z_beta_actual > z_beta]) else 'Model rejected') + ' for {}'.format(key))
        
        return

            # prob_alpha_error = 2*(1 - norm.cdf(mean_val * np.sqrt(N) / std_tr))

    def significance_histogram(self, key, save=False):
        
        dataset_deltas   = self.training_deltas[key]
        validation_deltas = self.validation_deltas[key] 
        fig, ax = plt.subplots(figsize=(4, 3))

        ax.hist(dataset_deltas, bins = len(dataset_deltas) // 5, density=True, color='red', label='training samples')

        # ax.hist(validation_deltas, bins = len(validation_deltas), density=True, color='blue', label='validation samples', alpha = 0.5)


        for i, val_delta in enumerate(validation_deltas):
            linestyle = 'dashed'
            color = 'b'
            if i == 0:  # Add label only to the first line to avoid redundant legend entries
                ax.axvline(val_delta, color=color, linestyle=linestyle, label='validation samples')
            else:
                ax.axvline(val_delta, color=color, linestyle=linestyle)

        # Generate data points for x-axis
        K = 2  * np.sqrt(2) 
        std_dataset = np.std(dataset_deltas)

        maxdiv = np.max(np.abs(dataset_deltas))
        x_range = np.linspace(- maxdiv, maxdiv + K * std_dataset)

        # (1) Normal distribution fit centered around zero
        normal_fit_0 = norm.pdf(x_range, loc=0, scale=std_dataset)
        ax.plot(x_range, normal_fit_0, color='k', linestyle='--',
                #  label='valid model fit'
                 )

        # (2) Normal distribution fit centered around K * std(dataset_deltas)
        normal_fit_K = norm.pdf(x_range, loc=K * std_dataset, scale=std_dataset)
        ax.plot(x_range, normal_fit_K, color='k', linestyle='--',
                #  label='invalid model fit'
                 )

        ax.set_xlabel('residual = ground truth - prediction')
        ax.set_ylabel('probability density')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=1, fancybox=True, shadow=True)
        ax.grid()
        plt.tight_layout()
        if save:
            plt.savefig('plots/significance_{}.svg'.format(key))
        else:
            plt.show()

    def get_derivatives(self, AOA, J, DELTA_E):
        
        data_div_alpha = np.stack((
            np.zeros(AOA.shape),
            np.ones(AOA.shape),
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            J,
            np.zeros(AOA.shape),
             DELTA_E,
            J * DELTA_E,
            2 * AOA,
            2 * AOA * DELTA_E,
            3 * AOA**2,
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            J**2,
            J * 2 * AOA,
            J**2 * 2 * AOA,
            J**2 * DELTA_E,
            J * 2 * AOA * DELTA_E,
            J**2 * 2 * AOA * DELTA_E,
        ), axis=-1)

        data_div_J = np.stack((
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.ones(AOA.shape),
            np.zeros(AOA.shape),
            AOA,
            DELTA_E,
            np.zeros(AOA.shape),
            AOA * DELTA_E,
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            2 * J,
            3 * J**2,
            2 * J * DELTA_E,
            2 * J * AOA,
            AOA **2,
            2 * J * AOA **2,
            2 * J * AOA * DELTA_E,
            AOA**2 * DELTA_E,
            2 * J * AOA**2 * DELTA_E,
        ), axis=-1)

        data_div_delta_e = np.stack((
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.ones(AOA.shape),
            np.zeros(AOA.shape),
            J,
            AOA,
            AOA * J,
            np.zeros(AOA.shape),
            AOA**2,
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            J**2,
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            np.zeros(AOA.shape),
            J**2 * AOA,
            J * AOA**2,
            J**2 * AOA**2,
        ), axis=-1)

        dalpha = {}
        dj = {}
        dde = {}
        for key in self.ground_truth.keys():
            dalpha[key] = np.tensordot(self.coefficients[key], data_div_alpha, axes=(0, -1))
            dj[key] = np.tensordot(self.coefficients[key], data_div_J, axes=(0, -1))
            dde[key] = np.tensordot(self.coefficients[key], data_div_delta_e, axes=(0, -1))
        
        return dalpha, dj, dde

    def plot_derivative_vs_alpha(self, key, derivative='alpha', save=False, AOA=np.linspace(-4, 7, 100), DELTA_E = [-10, 10], J=1.8):
        fig, ax = plt.subplots(figsize=(4, 3))

        colors = iter(plt.get_cmap('viridis')(np.linspace(0, 1, len(DELTA_E))))
        
        for delta_e in DELTA_E:
            c=next(colors)
            da, dj, dde = self.get_derivatives(AOA, np.full(AOA.shape, J), np.full(AOA.shape, delta_e))

            deriv = da[key] if derivative == 'alpha' else dj[key] if derivative == 'J' else dde[key]
            ax.plot(AOA, deriv, label=f'$\delta_e = {delta_e:.0f}$', color=c)


        var = 'alpha' if derivative == 'alpha' else 'J' if derivative == 'J' else 'delta_e'
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel(f'$d{key}/d{var}$')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        if save:
            plt.savefig('plots/{}_{}_vs_alpha.svg'.format(key, var))
        else:
            plt.show()

    def plot_derivative_vs_alpha_J(self, key, derivative='alpha', save=False, AOA=np.linspace(-4, 7, 100), DELTA_E = [-10, 10], J = np.linspace(1.6, 2.4, 100)):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = iter(plt.get_cmap('viridis')(np.linspace(0, 1, len(DELTA_E))))

        X, Y = np.meshgrid(AOA, J)
        
        for delta_e in DELTA_E:
            c=next(colors)
            da, dj, dde = self.get_derivatives(X, Y, np.full(X.shape, delta_e))
            deriv = da[key] if derivative == 'alpha' else dj[key] if derivative == 'J' else dde[key]
            ax.plot(X, Y, deriv, label=f'$\delta_e = {delta_e:.0f}$', color=c, alpha = 0.7)

        var = 'alpha' if derivative == 'alpha' else 'J' if derivative == 'J' else 'delta_e'
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$J$')
        ax.set_zlabel(f'$d{key}/d{var}$')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        if save:
            plt.savefig('plots/{}_{}_vs_alpha_J.svg'.format(key, var))
        else:
            plt.show()



    def plot_RSM_2D(self, key, reference_dataframe=None, validation_dataframe = None, save=False, DELTA_E=None, AOA=None, J=None, tolde = 1e-3, tolaoa=0.1, tolj=0.1, reference_label='Reference Points'):
        """
        plot a 2D slice of the response surface model
        you MUST set at least one of DELTA_E, AOA, or J to a value
        can add a reference data as a scatterplot
        """
        if all([DELTA_E is None, AOA is None, J is None]):
            raise ValueError('You must set at least one of DELTA_E, AOA, or J to a value')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mask = np.abs(self.data[3]-DELTA_E)<tolde if DELTA_E is not None else np.abs(self.data[1]-AOA)<tolaoa if AOA is not None else np.abs(self.data[2]-J)<tolj

        xvar = (self.data[1] if DELTA_E is not None else self.data[1] if J is not None else self.data[2])[mask]
        yvar = (self.data[2] if DELTA_E is not None else self.data[3] if J is not None else self.data[3])[mask]

        if reference_dataframe is not None and not isinstance(reference_dataframe, str):
            dataref = unpack_RSM_data(reference_dataframe)
            ref_mask = np.abs(dataref[3]-DELTA_E)<tolde if DELTA_E is not None else np.abs(dataref[1]-AOA)<tolaoa if AOA is not None else np.abs(dataref[2]-J)<tolj

            zvarref = reference_dataframe[key].to_numpy()[ref_mask]
            xvarref = (dataref[1] if DELTA_E is not None else dataref[1] if J is not None else dataref[2])[ref_mask]
            yvarref = (dataref[2] if DELTA_E is not None else dataref[3] if J is not None else dataref[3])[ref_mask]
            ax.scatter(xvarref, yvarref, zvarref, color='blue', marker='x', label=reference_label)

        elif reference_dataframe == 'self':
            xvarref = xvar
            yvarref = yvar
            zvarref = self.ground_truth[key][mask]
            ax.scatter(xvarref, yvarref, zvarref, color='blue', marker='x', label=reference_label)

        if validation_dataframe is not None and not isinstance(validation_dataframe, str):
            dataref = unpack_RSM_data(validation_dataframe)
            ref_mask = np.abs(dataref[3]-DELTA_E)<tolde if DELTA_E is not None else np.abs(dataref[1]-AOA)<tolaoa if AOA is not None else np.abs(dataref[2]-J)<tolj

            zvarref = reference_dataframe[key].to_numpy()[ref_mask]
            xvarref = (dataref[1] if DELTA_E is not None else dataref[1] if J is not None else dataref[2])[ref_mask]
            yvarref = (dataref[2] if DELTA_E is not None else dataref[3] if J is not None else dataref[3])[ref_mask]
            ax.scatter(xvarref, yvarref, zvarref, color='green', marker='o', label='Validation Points')

        elif validation_dataframe == 'self' and self.validation_dataframe is not None:
            val_mask = np.abs(self.validation_data[3]-DELTA_E)<tolde if DELTA_E is not None else np.abs(self.validation_data[1]-AOA)<tolaoa if AOA is not None else np.abs(self.validation_data[2]-J)<tolj
            xvarref = (self.validation_data[1] if DELTA_E is not None else self.validation_data[1] if J is not None else self.validation_data[2])[val_mask]
            yvarref = (self.validation_data[2] if DELTA_E is not None else self.validation_data[3] if J is not None else self.validation_data[3])[val_mask]
            zvarref = self.validation_ground_truth[key][val_mask]
            ax.scatter(xvarref, yvarref, zvarref, color='green', marker='o', label='Validation Points')
        elif self.validation_dataframe is None:
            print('Warning: attempting to plot validation dataset, but no validation points were specified')
        



            
        # Create a grid to interpolate onto
        grid_x, grid_y = np.linspace(xvar.min(), xvar.max(), 100), np.linspace(yvar.min(), yvar.max(), 100)
        X, Y = np.meshgrid(grid_x, grid_y)

        adj = (X, Y, np.full(X.shape, DELTA_E)) if DELTA_E is not None else (np.full(X.shape, AOA), X, Y) if AOA is not None else (X, np.full(X.shape, J), Y)
        Z = self._evaluate_from_AJD(self.coefficients[key], *adj)

        ax.plot_surface(X, Y, Z, color='red', alpha=0.3, label='Response Surface Model')


        xlabel = 'AoA' if DELTA_E is not None else 'AoA' if J is not None else 'J'
        ylabel = 'J' if DELTA_E is not None else 'DELTA_E' if J is not None else 'DELTA_E'

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(key)
        ax.legend()
        ax.grid()
        if save:
            plt.savefig('plots/RSM_2D_{}_{}_{}.svg'.format(key, xlabel, ylabel))
        else:
            plt.show()

    def plot_RSM_1D(self, key, reference_dataframe=None, validation_dataframe = None, save=False, DELTA_E=None, AOA=None, J=None, tolde = 1e-3, tolaoa=0.1, tolj=0.1, reference_label='Reference Points'):
        """
        plot a 1D slice of the response surface model
        you MUST set at least two of DELTA_E, AOA, or J to a value
        can add a reference data as a scatterplot
        """
        if sum([DELTA_E is None, AOA is None, J is None]) > 1:
            raise ValueError('You must set two of DELTA_E, AOA, or J to a value')

        fig, ax = plt.subplots(figsize=(4, 3))

        xvar = self.data[1] if (DELTA_E is not None and J is not None) else self.data[2] if (DELTA_E is not None and AOA is not None) else self.data[3]

        if reference_dataframe is not None and not isinstance(reference_dataframe, str):
            dataref = unpack_RSM_data(reference_dataframe)
            ref_mask = np.logical_and(
            np.abs(dataref[3] - DELTA_E) < tolde if DELTA_E is not None else np.full(dataref[3].shape, True),
            np.logical_and(
                np.abs(dataref[2] - J) < tolj if J is not None else np.full(dataref[3].shape, True),
                np.abs(dataref[1] - AOA) < tolaoa if AOA is not None else np.full(dataref[3].shape, True)
            )
            )
            zvarref = reference_dataframe[key].to_numpy()[ref_mask]
            xvarref = (dataref[1] if (DELTA_E is not None and J is not None) else dataref[2] if (DELTA_E is not None and AOA is not None) else dataref[3])[ref_mask]
            ax.scatter(xvarref, zvarref, color='blue', marker='x', label=reference_label)

        
        elif reference_dataframe == 'self':
            mask = np.logical_and(
            np.abs(self.data[3] - DELTA_E) < tolde if DELTA_E is not None else np.full(self.data[3].shape, True),
            np.logical_and(
                np.abs(self.data[2] - J) < tolj if J is not None else np.full(self.data[3].shape, True),
                np.abs(self.data[1] - AOA) < tolaoa if AOA is not None else np.full(self.data[3].shape, True)
            )
            )
            xvarref = xvar[mask]
            zvarref = self.ground_truth[key][mask]
            ax.scatter(xvarref, zvarref, color='blue', marker='x', label=reference_label)

        if validation_dataframe is not None and not isinstance(validation_dataframe, str):
            dataref = unpack_RSM_data(validation_dataframe)
            ref_mask = np.logical_and(
            np.abs(dataref[3] - DELTA_E) < tolde if DELTA_E is not None else np.full(dataref[3].shape, True),
            np.logical_and(
                np.abs(dataref[2] - J) < tolj if J is not None else np.full(dataref[3].shape, True),
                np.abs(dataref[1] - AOA) < tolaoa if AOA is not None else np.full(dataref[3].shape, True)
            )
            )
            zvarref = reference_dataframe[key].to_numpy()[ref_mask]
            xvarref = (dataref[1] if (DELTA_E is not None and J is not None) else dataref[2] if (DELTA_E is not None and AOA is not None) else dataref[3])[ref_mask]
            ax.scatter(xvarref, zvarref, color='green', marker='o', label='Validation Points')

        elif validation_dataframe == 'self' and self.validation_dataframe is not None:
            val_mask = np.logical_and(
            np.abs(self.validation_data[3] - DELTA_E) < tolde if DELTA_E is not None else np.full(self.validation_data[3].shape, True),
            np.logical_and(
                np.abs(self.validation_data[2] - J) < tolj if J is not None else np.full(self.validation_data[3].shape, True),
                np.abs(self.validation_data[1] - AOA) < tolaoa if AOA is not None else np.full(self.validation_data[3].shape, True)
            )
            )
            xvarref = (self.validation_data[1] if (DELTA_E is not None and J is not None) else self.validation_data[2] if (DELTA_E is not None and AOA is not None) else self.validation_data[3])[val_mask]
            zvarref = self.validation_ground_truth[key][val_mask]
            ax.scatter(xvarref, zvarref, color='green', marker='o', label='Validation Points')

        elif self.validation_dataframe is None:
            print('Warning: attempting to plot validation dataset, but no validation points were specified')


        # Create a grid to interpolate onto
        X = np.linspace(xvar.min(), xvar.max(), 100)
        adj = (X, np.full(X.shape, J) , np.full(X.shape, DELTA_E)) if (DELTA_E is not None and J is not None) else (np.full(X.shape, AOA), X , np.full(X.shape, DELTA_E)) if (DELTA_E is not None and AOA is not None) else (np.full(X.shape, AOA), np.full(X.shape, J), X)
        model_var = self._evaluate_from_AJD(self.coefficients[key], *adj)

        ax.plot(X, model_var, color='red', label='Response Surface Model')


        xlabel = 'AoA' if (DELTA_E is not None and J is not None) else 'J' if (DELTA_E is not None and AOA is not None) else 'DELTA_E'

        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.legend()
        ax.grid()
        plt.tight_layout()
        if save:
            plt.savefig('plots/RSM_1D_{}_{}.svg'.format(key, xlabel))
        else:
            plt.show()



if __name__ == "__main__":
    df = pd.read_csv('tunnel_data_unc/combined_data.txt', delimiter = '\t')

    df_RSM = df.loc[c.mask_RSM]
    df_validation = df.loc[c.mask_validation]

    rsm = ResponseSurfaceModel(df_RSM, df_validation)
    rsm.significance_histogram('CL')
    # coeff, res, loss = rsm.fit()
    # print(coeff, res)

    # for key in ['CL', 'CD', 'CMpitch']:
    #     rsm.plot_RSM(key=key, DELTA_E=-10, save=True, reference_dataframe='self', validation_dataframe='self')
    #     rsm.plot_RSM(key=key, AOA=7, save=True, reference_dataframe='self', validation_dataframe='self')
    #     rsm.plot_RSM(key=key, J=1.8, save=True, reference_dataframe='self', validation_dataframe='self')


    # print(rsm.predict(df))
