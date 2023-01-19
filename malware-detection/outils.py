import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
import datetime
import warnings

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn import tree # for decision tree models

# for model evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, average_precision_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization
import graphviz # for plotting decision tree graphs

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb


class Outils:
    def plot_prc(self, clf, y_true, y_score, title):
        class_names = clf.classes_
        n_classes = len(class_names)
        
        y_true = label_binarize(y_true, classes=class_names)
        y_score = label_binarize(y_score, classes=class_names)
        
        precision = dict()
        recall = dict()
        prc_auc = dict()
        
        # Plot all PRC curves
        plt.figure(dpi=600)
        lw = 2
        
        for i, class_name in enumerate(class_names):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
            prc_auc[i] = auc(recall[i], precision[i])
            plt.plot(recall[i], precision[i], lw=lw, label=class_name)
        
        # Compute micro-average PRC curve and PRC area
        recall["micro"], precision["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
        prc_auc["micro"] = auc(recall["micro"], precision["micro"])

        # First aggregate recall rates
        all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            mean_precision += np.interp(all_recall, recall[i], precision[i])

        # Finally average it and compute AUC
        mean_precision /= n_classes

        recall["macro"] = all_recall
        precision["macro"] = mean_precision
        prc_auc["macro"] = auc(recall["macro"], precision["macro"])

        plt.plot(recall["micro"], precision["micro"], label="micro-average PRC curve (area = {0:0.2f})".format(prc_auc["micro"]), color="deeppink", linestyle=":", linewidth=4,)

        plt.plot(recall["macro"], precision["macro"], label="macro-average PRC curve (area = {0:0.2f})".format(prc_auc["macro"]), color="navy", linestyle=":", linewidth=4,)
        
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.show()

    def plot_roc(self, clf, y_true, y_score, title):
        class_names = clf.classes_
        n_classes = len(class_names)
        
        y_true = label_binarize(y_true, classes=class_names)
        y_score = label_binarize(y_score, classes=class_names)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Plot all ROC curves
        plt.figure(dpi=600)
        lw = 2
        
        for i, class_name in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=lw, label=class_name)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]), color="deeppink", linestyle=":", linewidth=4,)

        plt.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]), color="navy", linestyle=":", linewidth=4,)
        
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.show()

    def plot_learning_curve(
            self,
            estimator,
            title,
            X,
            y,
            ylim=None,
            cv=None,
            n_jobs=None,
            scoring=None,
            train_sizes=np.linspace(0.1, 1.0, 5),
        ):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        scoring : str or callable, default=None
            A str (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        fig = plt.figure(dpi=600)

        grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
        axe_1 = plt.subplot(grid[0,:])
        axe_2 = plt.subplot(grid[1,0])
        axe_3 = plt.subplot(grid[1,1])
        axes = [axe_1, axe_2, axe_3] 

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
        plt.show()

    def performance_report(self, model, X, y_true, target_names=None):
        y_pred = model.predict(X)
        
        score_tr = model.score(X, y_true)
        print('Accuracy Score: ', score_tr)
        print("")
        bal_acc_te = balanced_accuracy_score(y_true, y_pred)
        print('Balanced Accuracy: ', bal_acc_te)
        print("")
        mat_corr_coeff_te = matthews_corrcoef(y_true, y_pred)
        print('Matthews Correlation Coefficient: ', mat_corr_coeff_te)
        print("")

        #auprc_te = average_precision_score(y_true, y_pred[:,1], pos_label=1)
        #print('Average Precision Score: ', auprc_te)
        #plot_precision_recall_curve(y_true, y_pred, clf.n_classes_)
        # Look at classification report to evaluate the model
        print(classification_report(y_true, y_pred, target_names=target_names))
        print('--------------------------------------------------------')

        print('************ Precision Recall Curve ************')
        self.plot_prc(model, y_true, y_pred, title='Precision Recall Curve')

        print('************ Receiver Operating Characteristic Curve ************')
        self.plot_roc(model, y_true, y_pred, title='Receiver Operating Characteristic Curve')
        
        # confusion matrix
        #print(confusion_matrix(y_test, pred_labels_te))
        matrix = plot_confusion_matrix(model, X, y_true, cmap=plt.cm.Blues, display_labels=target_names)
        matrix.ax_.set_title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=90)
        plt.show()
        print("")

    #allow logloss and classification error plots for each iteraetion of xgb model
    def plot_compare(metrics, eval_results, epochs):
        for m in metrics:
            test_score = eval_results['val'][m]
            train_score = eval_results['train'][m]
            rang = range(0, epochs)
            plt.rcParams["figure.figsize"] = [6,6]
            plt.plot(rang, test_score, "c", label="Val")
            plt.plot(rang, train_score, "orange", label="Train")
            title_name = m + " plot"
            plt.title(title_name)
            plt.xlabel('Iterations')
            plt.ylabel(m)
            lgd = plt.legend()
            plt.show()