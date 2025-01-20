\section{Introduction}
Poisson image editing is a technique used to seamlessly blend an object or texture from a source image into a target image. This project involves implementing the algorithm in Python and demonstrating its effectiveness with sample images.

\section{Project Overview}
\subsection{Source Code}
The primary implementation of the Poisson image editing algorithm is contained in the \texttt{poisson\_image\_editing.py} script. This script performs the necessary computations to blend the source and target images seamlessly. The implementation of the code has referenced Zhou's work.[\cite{trinkle23897}](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing)

\subsection{Images}
The project uses the following images:
\begin{itemize}
    \item \texttt{source.png}: The source image containing the object or texture to be blended.
    \item \texttt{target.png}: The target image into which the source image will be blended.
    \item \texttt{result.png}: The resulting image after applying the Poisson image editing algorithm.
\end{itemize}

\subsection{Running the Code}
To run the code, execute the following command in your terminal:
\begin{verbatim}
python poisson_image_editing.py -s <source
image> -t <target image> [-m <mask image>]
\end{verbatim}

Ensure that the \texttt{<source image>} and \texttt{<target image>} images are placed in the same directory with \texttt{poisson\_image\_editing.py}.

\begin{itemize}
    \item If the mask image is not specified, a window will pop up for you to draw the mask on the source image:
    
    \textbf{The green region will be used as the mask. Press `s` to save the result, press `r` to reset. }
    \item After the mask is defined, a window will pop up for you to adjust the mask position on the target image: 
    The mask corresponds to the region of source image that will be copied, you can move the mask to put the copied part into desired location in the target image. \textbf{Press `s` to save the result, press `r` to reset. }
\end{itemize}

Then the Poisson image editing process will start. The blended image will be named as \texttt{target\_result.png}, in the same directory as the source image. 
