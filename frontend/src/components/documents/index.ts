// Document Components Export
export { default as FileUploader } from './FileUploader';
export type { UploadFile } from './FileUploader';
export { default as UploadOptions } from './UploadOptions';
// WS-3: re-export the Modality string-union so callers can type their
// ``modalityOverride`` arrays without importing the implementation file.
export type { Modality } from './UploadOptions';
