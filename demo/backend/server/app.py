# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
    APP_ROOT,  # Ensure APP_ROOT is imported if used for path construction here
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import (
    Flask,
    make_response,
    Request,
    request,
    Response,
    send_from_directory,
    jsonify,
)
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView
from pathlib import Path


logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, **args)
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        for chunk in inference_api.propagate_in_video(request=request):
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()


@app.route("/session/<session_id>/save_last_propagation_masks", methods=["POST"])
def save_last_propagation_masks_route(session_id: str) -> Response:
    try:
        saved_file_path = inference_api.save_masks_from_last_propagation(session_id)
        return make_response(
            jsonify(
                {
                    "message": "All masks from the last propagation run saved successfully.",
                    "file_path": saved_file_path,
                }
            ),
            200,
        )
    except ValueError as e:  # Specific error for "no data to save" or "session not found" from predictor
        logger.error(f"Error saving propagation masks for session {session_id}: {e}")
        if "Cannot find session" in str(e) or "No propagation run data found" in str(e):
            return make_response(
                jsonify({"error": str(e)}), 404
            )  # Not Found or Bad Request
        return make_response(
            jsonify({"error": str(e)}), 400
        )  # Bad Request for other ValueErrors
    except RuntimeError as e:  # Covers file save failure from predictor
        logger.error(
            f"Runtime error saving propagation masks for session {session_id}: {e}"
        )
        return make_response(jsonify({"error": str(e)}), 500)
    except Exception as e:
        logger.error(
            f"Unexpected error saving propagation masks for session {session_id}: {e}",
            exc_info=True,
        )
        return make_response(
            jsonify(
                {
                    "error": "An unexpected error occurred while saving the propagation masks."
                }
            ),
            500,
        )


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
