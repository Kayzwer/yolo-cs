using System.Drawing;


namespace YOLO
{
    public class YoloLabel
    {
        public int Id { get; set; }

        public string Name { get; set; }

        public Color Color { get; set; }

        public YoloLabel(int Id, string Name, Color Color)
        {
            this.Id = Id;
            this.Name = Name;
            this.Color = Color;
        }
    }
}
