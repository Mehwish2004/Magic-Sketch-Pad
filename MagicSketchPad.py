import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame, StringVar, OptionMenu, Toplevel, Listbox, ttk, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
import threading

class MagicSketchPad:
    def __init__(self, root):
        self.root = root
        self.root.title("Magic Sketch Pad - DIP Project")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)  # Enable resizing and maximization
        
        self.current_image_path = None
        self.current_frame = None
        self.original_pil_image = None
        self.processed_frames = {}
        self.effect_strength = 100
        self.brightness = 1.0
        self.contrast = 1.0
        self.selected_effect = StringVar(value="Original")
        self.history = []
        self.history_index = -1
        self.processing = False
        self.control_panel_visible = True
        self.is_maximized = False
        
        self.setup_ui()
    
    def setup_ui(self):
        self.main_frame = Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.control_panel = Frame(self.main_frame, bg="#e0e0e0", bd=2, relief=tk.RAISED)
        self.control_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        Button(self.control_panel, text="🗖", command=self.toggle_maximize, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        Button(self.control_panel, text="☰", command=self.toggle_control_panel, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        Button(self.control_panel, text="Load Image", command=self.load_image, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        Button(self.control_panel, text="Save Image", command=self.save_current, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        effect_label = Label(self.control_panel, text="Select Effect:", bg="#e0e0e0")
        effect_label.pack(side=tk.LEFT, padx=5)
        effects = ["Original", "Sketch", "Cartoon", "Glow", "Oil", "Watercolor", 
                   "Grayscale", "Sepia", "Invert", "Blur", "Sharpness", "Dreamy Fade"]
        self.effect_menu = OptionMenu(self.control_panel, self.selected_effect, *effects, command=self.apply_effect)
        self.effect_menu.pack(side=tk.LEFT, padx=5)
        
        self.display_panel = Frame(self.main_frame, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.display_panel.grid(row=1, column=0, sticky="nsew", padx=5)
        
        self.history_panel = Frame(self.main_frame, bg="#e0e0e0", bd=2, relief=tk.RAISED)
        self.history_panel.grid(row=1, column=1, sticky="ns", padx=5)
        Label(self.history_panel, text="Effect History", bg="#e0e0e0", font=("Arial", 12, "bold")).pack(pady=5)
        self.history_list = Listbox(self.history_panel, width=30, height=20)
        self.history_list.pack(padx=5, pady=5)
        self.history_list.bind('<<ListboxSelect>>', self.select_history)
        
        self.preview_window = None
        self.preview_label = None
        
        self.progress = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        
        self.status_label = Label(self.main_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#d0d0d0")
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="ew")
        
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
    
    def toggle_maximize(self):
        if self.is_maximized:
            self.root.state('normal')
            self.control_panel.winfo_children()[0].config(text="🗖")
            self.is_maximized = False
        else:
            self.root.state('zoomed')
            self.control_panel.winfo_children()[0].config(text="🗗")
            self.is_maximized = True
    
    def toggle_control_panel(self):
        if self.control_panel_visible:
            self.control_panel.grid_remove()
            self.history_panel.grid_remove()
            self.control_panel.winfo_children()[1].config(text="➤")
        else:
            self.control_panel.grid()
            self.history_panel.grid()
            self.control_panel.winfo_children()[1].config(text="☰")
        self.control_panel_visible = not self.control_panel_visible
    
    def get_bg_color(self):
        return "#d0d0d0"
    
    def load_image(self):
        if self.processing:
            messagebox.showinfo("Processing", "Please wait until current processing is complete.")
            return
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if path:
            self.current_image_path = path
            self.original_pil_image = Image.open(path).convert("RGB")
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
            self.history = []
            self.history_index = -1
            self.processed_frames.clear()
            self.display_original_image()
            self.process_image()
    
    def display_original_image(self):
        if self.original_pil_image:
            original_cv = cv2.cvtColor(np.array(self.original_pil_image), cv2.COLOR_RGB2BGR)
            self.display_image(original_cv)
            self.status_label.config(text="Displaying original image")
    
    def process_image(self):
        if self.current_image_path and os.path.exists(self.current_image_path) and not self.processing:
            self.processing = True
            self.status_label.config(text="Processing...")
            self.progress['value'] = 0
            self.root.update()
            threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        self.current_frame = cv2.imread(self.current_image_path)
        if self.current_frame is None:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: Could not load image"))
            self.processing = False
            return

        max_dim = 600
        h, w = self.current_frame.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            self.current_frame = cv2.resize(self.current_frame, (int(w * scale), int(h * scale)))
        
        self.original_pil_image = self.original_pil_image.resize((self.current_frame.shape[1], self.current_frame.shape[0]), Image.LANCZOS)
        
        total_steps = 12
        step = 100 / total_steps
        
        self.processed_frames = {}
        self.processed_frames["Original"] = self.current_frame.copy()
        self.progress['value'] += step
        self.root.update()
        
        effect_method_map = {
            "Sketch": "apply_sketch_effect",
            "Cartoon": "apply_cartoon_effect",
            "Glow": "apply_glow_effect",
            "Oil": "apply_oil_painting_effect",
            "Watercolor": "apply_watercolor_effect",
            "Grayscale": "apply_grayscale_effect",
            "Sepia": "apply_sepia_effect",
            "Invert": "apply_invert_effect",
            "Blur": "apply_blur_effect",
            "Sharpness": "apply_sharpness_effect",
            "Dreamy Fade": "apply_dreamy_fade_effect"
        }
        
        for effect in effect_method_map:
            method_name = effect_method_map[effect]
            method = getattr(self, method_name)
            if effect in ["Sketch", "Cartoon", "Glow", "Oil", "Watercolor"]:
                self.processed_frames[effect] = method(self.current_frame)
            else:
                rgb_array = method(self.original_pil_image.copy())
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                self.processed_frames[effect] = bgr_array
            self.progress['value'] += step
            self.root.update()
        
        self.root.after(0, self.apply_effect, self.selected_effect.get())
        self.root.after(0, lambda: self.status_label.config(text="Processing complete"))
        self.processing = False
    
    def apply_effect(self, effect_name):
        if not self.processed_frames or effect_name not in self.processed_frames:
            return
        
        frame = self.processed_frames[effect_name].copy()
        cv2.putText(frame, effect_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(frame, effect_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1)
        self.display_image(frame)
        
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(effect_name)
        self.history_index += 1
        self.update_history_list()
        
        self.show_preview(frame)
    
    def display_image(self, cv_image):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_image)
        display_width = min(self.display_panel.winfo_width() - 20, 1200)
        display_height = min(self.display_panel.winfo_height() - 20, 800)
        if display_width > 100 and display_height > 100:
            pil_img.thumbnail((display_width, display_height), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        if hasattr(self, 'display_label'):
            self.display_label.destroy()
        self.display_label = Label(self.display_panel, image=tk_img, bg="#ffffff")
        self.display_label.image = tk_img
        self.display_label.pack(fill=tk.BOTH, expand=True)
    
    def show_preview(self, cv_image):
        if not self.preview_window:
            self.preview_window = Toplevel(self.root)
            self.preview_window.title("Effect Preview")
            self.preview_window.geometry("200x200")
            self.preview_window.protocol("WM_DELETE_WINDOW", self.close_preview)
            Label(self.preview_window, text="Real-time effect preview").pack()
            self.preview_label = Label(self.preview_window)
            self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_image)
        pil_img.thumbnail((150, 150), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=tk_img)
        self.preview_label.image = tk_img
    
    def close_preview(self):
        if self.preview_window:
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None
    
    def update_strength(self, val):
        self.effect_strength = int(val)
        if self.current_frame:
            self.process_image()
    
    def update_brightness(self, val):
        self.brightness = float(val)
        if self.original_pil_image:
            self.processed_frames["Brightness"] = self.apply_brightness_effect(self.original_pil_image.copy(), self.brightness)
            if self.selected_effect.get() == "Brightness":
                self.apply_effect("Brightness")
            self.show_preview(self.processed_frames["Brightness"])
    
    def update_contrast(self, val):
        self.contrast = float(val)
        if self.original_pil_image:
            self.processed_frames["Contrast"] = self.apply_contrast_effect(self.original_pil_image.copy(), self.contrast)
            if self.selected_effect.get() == "Contrast":
                self.apply_effect("Contrast")
            self.show_preview(self.processed_frames["Contrast"])
    
    def save_current(self):
        if not self.processed_frames or self.selected_effect.get() not in self.processed_frames:
            self.status_label.config(text="No processed image to save")
            return
        effect_name = self.selected_effect.get()
        save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            if os.path.exists(save_path):
                if not messagebox.askyesno("Overwrite File", f"File {os.path.basename(save_path)} exists. Overwrite?"):
                    self.status_label.config(text="Save cancelled")
                    return
            cv2.imwrite(save_path, self.processed_frames[effect_name])
            self.status_label.config(text=f"Saved {effect_name} to {os.path.basename(save_path)}")
    
    def update_history_list(self):
        self.history_list.delete(0, tk.END)
        for effect in self.history:
            self.history_list.insert(tk.END, effect)
        if self.history_index >= 0:
            self.history_list.selection_clear(0, tk.END)
            self.history_list.selection_set(self.history_index)
            self.history_list.activate(self.history_index)
    
    def select_history(self, event):
        selection = self.history_list.curselection()
        if selection:
            index = selection[0]
            self.history_index = index
            self.selected_effect.set(self.history[index])
            self.apply_effect(self.history[index])
    
    def apply_sketch_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_edges = cv2.Canny(gray_blur, 30, 100)
        laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        edges = cv2.addWeighted(canny_edges, 0.7, laplacian, 0.3, 0)
        ret, sketch = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)
        noise = np.zeros_like(gray)
        cv2.randu(noise, 0, 10)
        sketch = cv2.add(sketch, noise)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def apply_cartoon_effect(self, frame):
        smoothed = cv2.bilateralFilter(frame, 9, 75, 75)
        Z = smoothed.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        color = center[label.flatten()].reshape(frame.shape)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges_inv = cv2.bitwise_not(edges)
        cartoon = cv2.bitwise_and(color, color, mask=edges_inv)
        gray_shade = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        shade = cv2.bitwise_and(color, color, mask=gray_shade)
        cartoon = cv2.addWeighted(cartoon, 0.9, shade, 0.1, 0)
        return cartoon

    def apply_glow_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        glow1 = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
        glow2 = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        glow3 = cv2.GaussianBlur(edges.astype(np.float32), (25, 25), 0)
        soft_edges = cv2.addWeighted(cv2.addWeighted(glow1, 0.5, glow2, 0.3, 0), 0.8, glow3, 0.2, 0)
        glow = np.zeros_like(frame).astype(np.float32)
        glow[:, :, 1] = soft_edges * 1.0
        glow[:, :, 0] = soft_edges * 0.7
        glow[:, :, 2] = soft_edges * 0.4
        glow = np.clip(glow, 0, 255).astype(np.uint8)
        return cv2.addWeighted(frame, 0.8, glow, 0.6, 0)

    def apply_oil_painting_effect(self, frame):
        oil = cv2.bilateralFilter(frame, 15, 80, 80)
        oil = cv2.medianBlur(oil, 5)
        oil = (oil // 16) * 16
        return oil

    def apply_watercolor_effect(self, frame):
        watercolor = cv2.edgePreservingFilter(frame, flags=cv2.NORMCONV_FILTER, sigma_s=60, sigma_r=0.4)
        watercolor = cv2.medianBlur(watercolor, 3)
        hsv = cv2.cvtColor(watercolor, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        noise = np.zeros(frame.shape[:2], np.uint8)
        cv2.randu(noise, 0, 15)
        texture = cv2.GaussianBlur(noise, (21, 21), 0)
        texture_rgb = cv2.merge([texture, texture, texture])
        watercolor = cv2.addWeighted(watercolor, 0.9, texture_rgb, 0.1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        for i in range(3):
            watercolor[:, :, i] = np.clip(watercolor[:, :, i] * (1.0 - 0.05 * edges / 255.0), 0, 255).astype(np.uint8)
        return watercolor

    def apply_grayscale_effect(self, img):
        gray_img = img.convert("L").convert("RGB")
        return np.array(gray_img)

    def apply_sepia_effect(self, img):
        sepia = img.copy()
        pixels = sepia.load()
        for y in range(sepia.height):
            for x in range(sepia.width):
                r, g, b = pixels[x, y]
                tr = int(0.393*r + 0.769*g + 0.189*b)
                tg = int(0.349*r + 0.686*g + 0.168*b)
                tb = int(0.272*r + 0.534*g + 0.131*b)
                pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
        return np.array(sepia)

    def apply_invert_effect(self, img):
        inverted = ImageOps.invert(img)
        return np.array(inverted)

    def apply_blur_effect(self, img):
        blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
        return np.array(blurred)

    def apply_sharpness_effect(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        sharpened = enhancer.enhance(2.0)
        return np.array(sharpened)

    def apply_dreamy_fade_effect(self, img):
        img = img.copy().convert("RGB")
        img = ImageEnhance.Color(img).enhance(0.6)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        overlay = Image.new("RGB", img.size, (255, 228, 225))
        img = Image.blend(img, overlay, alpha=0.2)
        return np.array(img)

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicSketchPad(root)
    root.mainloop()