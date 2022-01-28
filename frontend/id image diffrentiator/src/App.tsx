import { useState, FC } from "react";
import axios from "axios";
import "./App.css";

interface file {
    imagetype?: String;
    filename?: String;
    message?: String;
}

const App: FC = () => {
    const [fileUploaded, setFileUploaded] = useState<File>();
    const [response, setResponse] = useState<file>();

    //for error handling
    const [error, setError] = useState("");

    const handleImageChange = function (
        e: React.ChangeEvent<HTMLInputElement>
    ) {
        const fileList = e.target.files;

        if (!fileList) return;

        setFileUploaded(fileList[0]);
    };

    const uploadFile = function (
        e: React.MouseEvent<HTMLSpanElement, MouseEvent>
    ) {
        if (fileUploaded) {
            const formData = new FormData();
            formData.append("file", fileUploaded, fileUploaded.name);
            console.log(formData);
            axios
                .post("http://127.0.0.1:8000/file/upload/", formData)
                .then((res) => {
                    let data = res.data;
                    console.log(data);
                    setResponse(data);
                    setError("");
                    setFileUploaded(undefined)
                })
                .catch((err) => {
                    alert(err);
                    setError(err);
                });
        }
    };

    return (
        <div className="container-fluid vh-100 d-flex justify-content-center align-items-center py-5 bg-light text-dark">
            <div className="py-5">
                <p>Image processing takes a couple minutes</p>
                <div className="mb-3">
                    <label
                        htmlFor="exampleFormControlInput1"
                        className="form-label fw-bold"
                    >
                        Upload Passport or ID image
                    </label>
                    <input
                        type="file"
                        className="form-control"
                        id="photo"
                        name="photo"
                        multiple={false}
                        onChange={handleImageChange}
                    />
                </div>
                <div className="mb-3">
                    <button onClick={uploadFile} className={response?.imagetype ? (
                        "btn btn-success"
                    ) : (
                        "btn btn-danger"
                    )} >
                        {response?.imagetype ? (
                            "Processed Picture"
                        ) : (
                            "Process Picture"
                        )}
                    </button>
                </div>
                <div className="text-secondary">
                    {response?.imagetype ? (
                        <div>
                            <p>Image uploaded "{response.filename}" is</p>
                            <strong className="text-danger">{response.imagetype}.</strong>
                        </div>
                    ) : (
                        <p>{response?.message}</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default App;
