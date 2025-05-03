import streamlit as st
import os
from PIL import Image


st.title("Parkinson detection using Detectron2")
st.divider()
st.subheader('Metrics Graphs')
st.divider()

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Metrics Graphs", "Model Predictions", "Confusion Matrix"])

# Tab 1: Metrics Graphs
with tab1:
    st.subheader('Training Metrics')
    st.divider()

    GRAPH_DIR = 'output_detectron'


    @st.cache_data(ttl=3600)
    def get_graph_images():
        """Get all graph images from the output directory with caching"""
        images = []
        valid_extensions = ('.png', '.jpg', '.jpeg')

        if os.path.exists(GRAPH_DIR):
            for file in sorted(os.listdir(GRAPH_DIR)):
                if file.lower().endswith(valid_extensions):
                    try:
                        img_path = os.path.join(GRAPH_DIR, file)
                        images.append((file, Image.open(img_path)))
                    except Exception as e:
                        st.error(f'Error loading {file}: {str(e)}')
        else:
            st.error(f'Directory not found: {GRAPH_DIR}')

        return images


    graph_images = get_graph_images()

    if graph_images:
        # Sidebar controls
        st.sidebar.header('Graph Display Options')

        # Single graph selection
        selected_graph = st.sidebar.selectbox(
            'Choose a graph to display:',
            options=[img[0] for img in graph_images],
            index=0
        )

        # Multiple graph comparison
        compare_graphs = st.sidebar.multiselect(
            'Compare multiple graphs:',
            options=[img[0] for img in graph_images],
            default=[graph_images[0][0]]
        )

        # Graph browser slider
        graph_index = st.sidebar.slider(
            'Browse graphs:',
            0, len(graph_images) - 1, 0,
            help='Use slider to quickly browse through graphs'
        )

        # Display selected single graph
        st.header("Selected Graph")
        selected_img = next(img for name, img in graph_images if name == selected_graph)
        st.image(selected_img,
                 caption=f'Selected: {selected_graph}',
                 use_container_width=True)

        # Display comparison view if multiple graphs selected
        if len(compare_graphs) > 1:
            st.header('Comparison View')
            cols = st.columns(len(compare_graphs))
            for col, img_name in zip(cols, compare_graphs):
                img = next(img for name, img in graph_images if name == img_name)
                col.image(img, caption=img_name, use_container_width=True)

        # Quick browse via slider
        st.header('Quick Browse')
        slider_img = graph_images[graph_index][1]
        st.image(slider_img,
                 caption=f'Graph {graph_index + 1}/{len(graph_images)}: {graph_images[graph_index][0]}',
                 use_container_width=True)

    else:
        st.warning('No graph images found in the output directory')

# Tab 2: Model Predictions with Bounding Boxes
with tab2:
    st.subheader('Model Predictions with Bounding Boxes')
    st.divider()

    RESULTS_IMAGES_DIRECTORY = 'Results'


    @st.cache_data(ttl=3600)
    def get_prediction_images():
        """Get all prediction images from Results directory"""
        images = []
        valid_extensions = ('.png', '.jpg', '.jpeg')

        if os.path.exists(RESULTS_IMAGES_DIRECTORY):
            for file in sorted(os.listdir(RESULTS_IMAGES_DIRECTORY)):
                if file.lower().endswith(valid_extensions):
                    try:
                        img_path = os.path.join(RESULTS_IMAGES_DIRECTORY, file)
                        images.append((file, Image.open(img_path)))
                    except Exception as e:
                        st.error(f'Error loading {file}: {str(e)}')
        else:
            st.error(f'Directory not found: {RESULTS_IMAGES_DIRECTORY}')

        return images


    prediction_images = get_prediction_images()

    if prediction_images:
        # Sidebar controls for predictions
        st.sidebar.header('Prediction Display Options')

        # Single prediction selection
        selected_prediction = st.sidebar.selectbox(
            'Choose a prediction to display:',
            options=[img[0] for img in prediction_images],
            index=0
        )

        # Multiple prediction comparison
        compare_predictions = st.sidebar.multiselect(
            'Compare multiple predictions:',
            options=[img[0] for img in prediction_images],
            default=[prediction_images[0][0]]
        )

        # Prediction browser slider
        pred_index = st.sidebar.slider(
            'Browse predictions:',
            0, len(prediction_images) - 1, 0,
            help='Use slider to quickly browse through predictions'
        )

        # Display selected prediction
        st.header("Selected Prediction")
        selected_pred = next(img for name, img in prediction_images if name == selected_prediction)
        st.image(selected_pred,
                 caption=f'Prediction: {selected_prediction}',
                 use_container_width=True)

        # Display comparison view if multiple predictions selected
        if len(compare_predictions) > 1:
            st.header('Comparison View')
            cols = st.columns(len(compare_predictions))
            for col, img_name in zip(cols, compare_predictions):
                img = next(img for name, img in prediction_images if name == img_name)
                col.image(img, caption=img_name, use_container_width=True)

        # Quick browse via slider
        st.header('Quick Browse')
        slider_pred = prediction_images[pred_index][1]
        st.image(slider_pred,
                 caption=f'Prediction {pred_index + 1}/{len(prediction_images)}: {prediction_images[pred_index][0]}',
                 use_container_width=True)

    else:
        st.warning('No prediction images found in the Results directory')

# Tab 3: Confusion Matrix
with tab3:
    st.subheader('Model Performance')
    st.divider()

    DETECTRON_CM = 'Detectron_confusion_matrix.png'

    if os.path.exists(DETECTRON_CM):
        st.image(DETECTRON_CM, caption='Confusion Matrix', use_container_width=True)
    else:
        st.error(f'Confusion matrix image not found: {DETECTRON_CM}')

    st.divider()

