import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.zip.ZipFile;
import javax.swing.*;

public class Dnn {

    static final String MODEL_PATH = "models/mnist-model.bin";
    static final String PERSONAL_MODEL_PATH = "models/mnist-model-personal.bin";
    static final String SAMPLES_PATH = "models/my-samples.bin";
    static final String DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz";

    // ---------- Matematik ----------

    // Ham sayıları, toplamı 1 olan olasılıklara çevirir
    static double[] softmax(double[] z) {
        double[] p = new double[z.length];
        double sum = 0.0;
        for (int i = 0; i < z.length; i++) {
            p[i] = Math.exp(z[i]);
            sum += p[i];
        }
        for (int k = 0; k < p.length; k++) {
            p[k] /= sum;
        }
        return p;
    }

    // En büyük değerin indeksi
    static int argMax(double[] a) {
        double max = a[0];
        int ind = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > max) {
                max = a[i];
                ind = i;
            }
        }
        return ind;
    }

    static double crossEntropy(double[] p, double[] target) {
        double lossSum = 0.0;
        for (int k = 0; k < target.length; k++) {
            lossSum -= (target[k] * Math.log(p[k]));
        }
        return lossSum;
    }

    // ---------- Ağ ----------

    // Bir katman = birkaç nöron. Her nöronun her girişe bir ağırlığı, bir de bias'ı var.
    static class Layer {
        double[][] weights; // weights[nöron][giriş]
        double[] biases; // biases[nöron]
        double[] lastInput;
        double[] lastOutput;
        boolean relu; // true: gizli katman (ReLU), false: çıkış katmanı (softmax)

        // He başlatma: sinyalin boyutunu katmanlar boyunca dengede tutar
        Layer(int inputs, int neurons, Random rnd, boolean relu) {
            weights = new double[neurons][inputs];
            biases = new double[neurons];
            this.relu = relu;
            for (int n = 0; n < neurons; n++) {
                for (int i = 0; i < inputs; i++) {
                    weights[n][i] = rnd.nextGaussian() * Math.sqrt(2.0 / inputs);
                }
            }
        }

        // Her nöron: (ağırlık * giriş) toplamı + bias, sonra aktivasyon
        double[] forward(double[] x) {
            double[] out = new double[weights.length];
            for (int n = 0; n < weights.length; n++) {
                double sum = biases[n];
                for (int i = 0; i < x.length; i++) {
                    sum += weights[n][i] * x[i];
                }
                out[n] = relu ? Math.max(0, sum) : sum;
            }
            if (!relu) {
                out = softmax(out);
            }
            lastInput = x;
            lastOutput = out;
            return out;
        }

        // Zincir kuralı: kendini günceller, önceki katmanın sinyalini döndürür.
        // Softmax + cross-entropy ikilisinde çıkış katmanının eğim çarpanı sadeleşir.
        double[] backward(double[] grad, double lr) {
            double[] gradInput = new double[lastInput.length];
            for (int n = 0; n < grad.length; n++) {
                double o = lastOutput[n];
                double delta = relu ? (o > 0 ? grad[n] : 0) : grad[n];
                for (int i = 0; i < lastInput.length; i++) {
                    gradInput[i] += delta * weights[n][i];
                    weights[n][i] -= lr * delta * lastInput[i];
                }
                biases[n] -= lr * delta;
            }
            return gradInput;
        }
    }

    // Ağ = üst üste katmanlar. Bir katmanın çıkışı sonrakinin girişi olur.
    static class Network {
        Layer[] layers;

        // sizes = {784, 256, 128, 10} -> 784 giriş, iki gizli katman, 10 çıkış
        Network(int[] sizes, Random rnd) {
            layers = new Layer[sizes.length - 1];
            for (int i = 0; i < layers.length; i++) {
                layers[i] = new Layer(sizes[i], sizes[i + 1], rnd, i < layers.length - 1);
            }
        }

        double[] forward(double[] x) {
            for (Layer layer : layers) {
                x = layer.forward(x);
            }
            return x;
        }

        void train(double[] x, double[] target, double lr) {
            double[] out = forward(x);
            double[] grad = new double[out.length];
            for (int k = 0; k < out.length; k++) {
                grad[k] = (out[k] - target[k]);
            }

            for (int l = layers.length - 1; l >= 0; l--) {
                grad = layers[l].backward(grad, lr);
            }
        }

        // Dosyada yalnızca art arda sayılar var, neyin nereye ait olduğunu sıra belirler
        void save(String path) throws IOException {
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
                for (Layer layer : layers) {
                    for (int n = 0; n < layer.weights.length; n++) {
                        for (int i = 0; i < layer.weights[n].length; i++) {
                            out.writeDouble(layer.weights[n][i]);
                        }
                        out.writeDouble(layer.biases[n]);
                    }
                }
            }
        }

        void load(String path) throws IOException {
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
                for (Layer layer : layers) {
                    for (int n = 0; n < layer.weights.length; n++) {
                        for (int i = 0; i < layer.weights[n].length; i++) {
                            layer.weights[n][i] = in.readDouble();
                        }
                        layer.biases[n] = in.readDouble();
                    }
                }
            }
        }
    }

    static double loss(Network net, double[][] inputs, double[][] targets) {
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum = sum + crossEntropy(net.forward(inputs[i]), targets[i]);
        }
        sum = sum / inputs.length;
        return sum;
    }

    static double accuracy(Network net, double[][] inputs, double[][] targets) {
        int cnt = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] guess = net.forward(inputs[i]);
            if (argMax(guess) == argMax(targets[i])) {
                cnt++;
            }
        }
        return (double) cnt / inputs.length;
    }

    // ---------- MNIST ----------

    // .npz bir zip dosyası, içindeki .npy dosyalarının başlığını atlayıp ham baytları alıyoruz
    static byte[] readNpy(ZipFile zip, String name) throws IOException {
        try (InputStream in = zip.getInputStream(zip.getEntry(name))) {
            in.readNBytes(8);
            int headerLength = in.read() | (in.read() << 8);
            in.readNBytes(headerLength);
            return in.readAllBytes();
        }
    }

    // Java'da byte işaretlidir: & 0xFF ile 0-255 aralığına, / 255.0 ile 0-1 aralığına getiriyoruz
    static double[][] toImages(byte[] raw) {
        int imgCnt = raw.length / 784;
        double[][] img = new double[imgCnt][784];
        for (int i = 0; i < img.length; i++) {
            for (int j = 0; j < img[0].length; j++) {
                img[i][j] = (raw[i * 784 + j] & 0xFF) / 255.0;
            }
        }
        return img;
    }

    static double[][] toOneHot(byte[] raw) {
        double[][] newArr = new double[raw.length][10];
        for (int i = 0; i < raw.length; i++) {
            newArr[i][raw[i]] = 1;
        }
        return newArr;
    }

    static File findData() {
        File local = new File("data/mnist.npz");
        if (local.exists()) {
            return local;
        }
        return new File(System.getProperty("user.home"), ".keras/datasets/mnist.npz");
    }

    // Resmi rastgele döndürür, eğer, ölçekler ve kaydırır. Ağ böylece daha çok şekil görür.
    static double[] augment(double[] image, Random rnd) {
        double angle = Math.toRadians(-15 + 30 * rnd.nextDouble());
        double shear = -0.2 + 0.4 * rnd.nextDouble();
        double scaleX = 0.9 + 0.2 * rnd.nextDouble();
        double scaleY = 0.9 + 0.2 * rnd.nextDouble();
        double shiftX = -1 + 2 * rnd.nextDouble();
        double shiftY = -1 + 2 * rnd.nextDouble();
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);

        double[] out = new double[784];
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double x = col - 13.5 - shiftX;
                double y = row - 13.5 - shiftY;
                double u = (cos * x + sin * y) / scaleX;
                double v = (-sin * x + cos * y) / scaleY;
                u = u + shear * v + 13.5;
                v = v + 13.5;

                int i = (int) Math.floor(u);
                int j = (int) Math.floor(v);
                if (i < 0 || j < 0 || i >= 27 || j >= 27) {
                    continue;
                }
                double fu = u - i;
                double fv = v - j;
                out[row * 28 + col] = image[j * 28 + i] * (1 - fu) * (1 - fv)
                        + image[j * 28 + i + 1] * fu * (1 - fv)
                        + image[(j + 1) * 28 + i] * (1 - fu) * fv
                        + image[(j + 1) * 28 + i + 1] * fu * fv;
            }
        }
        return out;
    }

    // ---------- Çizimi ağa hazırlama ----------

    // MNIST'in hazırlanış yöntemini taklit eder: rakamı 20x20 hücreye sığdırır, kütle merkezini ortalar
    static BufferedImage normalize(BufferedImage img) {
        int minX = 280, minY = 280, maxX = -1, maxY = -1;
        double mass = 0.0, sumX = 0.0, sumY = 0.0;

        for (int x = 0; x < 280; x++) {
            for (int y = 0; y < 280; y++) {
                int v = img.getRaster().getSample(x, y, 0);
                if (v > 0) {
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                    mass += v;
                    sumX += v * x;
                    sumY += v * y;
                }
            }
        }
        if (mass == 0.0) {
            return img;
        }
        double scale = 200.0 / Math.max(maxX - minX + 1, maxY - minY + 1);
        BufferedImage out = new BufferedImage(280, 280, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.translate(140, 140);
        g.scale(scale, scale);
        g.translate(-sumX / mass, -sumY / mass);
        g.drawImage(img, 0, 0, null);
        g.dispose();
        return out;
    }

    // 280x280 tuvali 28x28'e indirir: her 10x10'luk karenin ortalaması tek piksel olur
    static double[] toInput(BufferedImage img) {
        double[] input = new double[784];
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double sum = 0.0;
                for (int y = 0; y < 10; y++) {
                    for (int x = 0; x < 10; x++) {
                        sum += img.getRaster().getSample(col * 10 + x, row * 10 + y, 0);
                    }
                }
                input[row * 28 + col] = sum / 100 / 255.0;
            }
        }
        return input;
    }

    // ---------- Kişisel örnekler ----------

    // Yeni örneği öğretirken araya MNIST katıyoruz ki ağ eski bilgisinden kopmasın
    static void learnFrom(Network net, ArrayList<double[]> myX, ArrayList<double[]> myY, double[][] mnistX,
            double[][] mnistY, Random rnd) {
        for (int step = 0; step < 200; step++) {
            int k = rnd.nextBoolean() ? myX.size() - 1 : rnd.nextInt(myX.size());
            net.train(augment(myX.get(k), rnd), myY.get(k), 0.003);
            for (int r = 0; r < 5; r++) {
                int i = rnd.nextInt(mnistX.length);
                net.train(augment(mnistX[i], rnd), mnistY[i], 0.001);
            }
        }
    }

    static void appendSample(double[] x, int label) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(SAMPLES_PATH, true)))) {
            for (double v : x) {
                out.writeDouble(v);
            }
            out.writeInt(label);
        }
    }

    static void loadSamples(ArrayList<double[]> myX, ArrayList<double[]> myY) throws IOException {
        File file = new File(SAMPLES_PATH);
        if (!file.exists()) {
            return;
        }
        int count = (int) (file.length() / (784 * 8 + 4));
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            for (int c = 0; c < count; c++) {
                double[] x = new double[784];
                for (int i = 0; i < 784; i++) {
                    x[i] = in.readDouble();
                }
                double[] y = new double[10];
                y[in.readInt()] = 1;
                myX.add(x);
                myY.add(y);
            }
        }
    }

    // ---------- Pencere ----------

    static void openWindow(Network net, double[][] mnistX, double[][] mnistY) throws IOException {
        BufferedImage canvas = new BufferedImage(280, 280, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D pen = canvas.createGraphics();
        pen.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        pen.setStroke(new BasicStroke(22, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

        JLabel result = new JLabel("Bir rakam çiz", SwingConstants.CENTER);
        result.setFont(new Font("SansSerif", Font.BOLD, 22));

        JPanel board = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(canvas, 0, 0, null);
            }
        };
        board.setPreferredSize(new Dimension(280, 280));

        MouseAdapter mouse = new MouseAdapter() {
            int lastX, lastY;

            @Override
            public void mousePressed(MouseEvent e) {
                lastX = e.getX();
                lastY = e.getY();
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                pen.setColor(Color.WHITE);
                pen.drawLine(lastX, lastY, e.getX(), e.getY());
                lastX = e.getX();
                lastY = e.getY();
                board.repaint();

                double[] p = net.forward(toInput(normalize(canvas)));
                int guess = argMax(p);
                result.setText(String.format("Ağ: %d   güven: %.3f", guess, p[guess]));
            }
        };
        board.addMouseListener(mouse);
        board.addMouseMotionListener(mouse);

        JButton clear = new JButton("Temizle");
        clear.addActionListener(e -> {
            pen.setColor(Color.BLACK);
            pen.fillRect(0, 0, 280, 280);
            board.repaint();
            result.setText("Bir rakam çiz");
        });

        ArrayList<double[]> myX = new ArrayList<>();
        ArrayList<double[]> myY = new ArrayList<>();
        loadSamples(myX, myY);
        Random learnRnd = new Random();

        JPanel digits = new JPanel(new GridLayout(2, 5));
        for (int d = 0; d < 10; d++) {
            int digit = d;
            JButton button = new JButton(String.valueOf(digit));
            button.addActionListener(e -> {
                double[] input = toInput(normalize(canvas));
                if (Arrays.stream(input).sum() == 0.0) {
                    return;
                }
                double[] target = new double[10];
                target[digit] = 1;
                myX.add(input);
                myY.add(target);
                learnFrom(net, myX, myY, mnistX, mnistY, learnRnd);
                try {
                    net.save(PERSONAL_MODEL_PATH);
                    appendSample(input, digit);
                } catch (IOException ex) {
                    result.setText("Kaydedilemedi");
                    return;
                }
                double[] p = net.forward(input);
                int guess = argMax(p);
                result.setText(String.format("Öğrendi: %d   Ağ: %d (%.3f)", digit, guess, p[guess]));
            });
            digits.add(button);
        }

        JPanel bottom = new JPanel(new BorderLayout());
        bottom.add(new JLabel("Yanlış bildiyse doğrusuna bas:", SwingConstants.CENTER), BorderLayout.NORTH);
        bottom.add(digits, BorderLayout.CENTER);
        bottom.add(clear, BorderLayout.SOUTH);

        JFrame frame = new JFrame("Rakam Tanı");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(result, BorderLayout.NORTH);
        frame.add(board, BorderLayout.CENTER);
        frame.add(bottom, BorderLayout.SOUTH);
        frame.pack();
        frame.setResizable(false);
        frame.setVisible(true);
    }

    // ---------- Değerlendirme ----------

    static void printDigit(double[] image) {
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                System.out.print(image[row * 28 + col] > 0.5 ? "#" : ".");
            }
            System.out.println();
        }
    }

    static void evaluate(Network net, double[][] testX, double[][] testY) {
        System.out.printf("test: %.4f%n", accuracy(net, testX, testY));
        int shown = 0;
        int wrong = 0;
        int[][] confusion = new int[10][10];
        for (int i = 0; i < testX.length; i++) {
            double[] p = net.forward(testX[i]);
            int guess = argMax(p);
            int truth = argMax(testY[i]);
            confusion[truth][guess]++;
            if (guess != truth) {
                wrong++;
                if (shown < 3) {
                    System.out.printf("%ndoğru: %d  ağ: %d  güven: %.3f%n", truth, guess, p[guess]);
                    printDigit(testX[i]);
                    shown++;
                }
            }
        }
        System.out.println();
        for (int t = 0; t < confusion.length; t++) {
            for (int g = 0; g < confusion[0].length; g++) {
                System.out.printf("%5d", confusion[t][g]);
            }
            System.out.println();
        }
        System.out.println("toplam yanlış: " + wrong + " / " + testX.length);
    }

    public static void main(String[] args) throws IOException {
        File dataFile = findData();
        if (!dataFile.exists()) {
            System.err.println("MNIST bulunamadı. İndirmek için:");
            System.err.println("  mkdir -p data && curl -L -o data/mnist.npz " + DATA_URL);
            System.exit(1);
        }
        new File("models").mkdirs();

        double[][] allX, allY, testX, testY;
        try (ZipFile zip = new ZipFile(dataFile)) {
            allX = toImages(readNpy(zip, "x_train.npy"));
            allY = toOneHot(readNpy(zip, "y_train.npy"));
            testX = toImages(readNpy(zip, "x_test.npy"));
            testY = toOneHot(readNpy(zip, "y_test.npy"));
        }

        double[][] trainX = Arrays.copyOfRange(allX, 0, 50000);
        double[][] trainY = Arrays.copyOfRange(allY, 0, 50000);
        double[][] valX = Arrays.copyOfRange(allX, 50000, 60000);
        double[][] valY = Arrays.copyOfRange(allY, 50000, 60000);

        Network net = new Network(new int[] { 784, 256, 128, 10 }, new Random(42));

        if (new File(PERSONAL_MODEL_PATH).exists()) {
            net.load(PERSONAL_MODEL_PATH);
            System.out.println("kişisel model yüklendi.");
        } else if (new File(MODEL_PATH).exists()) {
            net.load(MODEL_PATH);
            System.out.println("model dosyadan yüklendi.");
        } else {
            System.out.println("Kayıtlı model yok, eğitim başlıyor (birkaç dakika sürer).");
            Random augRnd = new Random(1);
            for (int epoch = 1; epoch <= 20; epoch++) {
                double lr = epoch > 15 ? 0.001 : 0.01;
                for (int i = 0; i < trainX.length; i++) {
                    net.train(augment(trainX[i], augRnd), trainY[i], lr);
                }
                System.out.printf("epoch %d  val loss: %.4f  val: %.4f%n", epoch, loss(net, valX, valY),
                        accuracy(net, valX, valY));
            }
            net.save(MODEL_PATH);
            System.out.println("model kaydedildi.");
        }

        if (args.length > 0 && args[0].equals("--eval")) {
            evaluate(net, testX, testY);
            return;
        }

        SwingUtilities.invokeLater(() -> {
            try {
                openWindow(net, trainX, trainY);
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        });
    }
}
